//! `LogitStore`: a captured set of next-token distributions, and its on-disk format.
//!
//! This type is the seam that keeps the eval harness from being hostage to any one runtime. The
//! per-layer sensitivity loop needs logits; where they come from is negotiable:
//!
//! - **llama.cpp** can produce them for architectures it supports, via its own KL-divergence base
//!   file — but that file is a llama.cpp-internal format we consume through its CLI, not parse.
//! - **An in-process forward pass** (candle) would be ideal, and is what §3.6 assumes.
//! - **An external server** — dump logits from vLLM on the box that can actually run the model,
//!   copy the file over, diff here.
//!
//! That third path is not a consolation prize. For a model whose attention is novel enough that
//! neither llama.cpp nor candle implements it, it is the *only* path, and building the harness
//! around a portable capture file rather than around an in-process forward pass is what makes the
//! difference between "eval is blocked on an architecture port" and "eval needs a 20-line dump
//! script on the serving box". [`LogitStore::write_to`] / [`LogitStore::read`] are that file, and
//! [`LogitStore::from_dense_f32`] ingests a bare fp32 dump for the shortest possible script.
//!
//! ## Format (`RQLG`, v1, little-endian)
//!
//! ```text
//! magic "RQLG" | u32 version | u32 vocab | u32 top_k (0 = dense) | u64 n_rows | u64 flags
//! u32 × n_rows   target token ids (u32::MAX = "no target at this position")
//! rows:
//!   dense : f32 × vocab
//!   sparse: u32 × top_k ids | f32 × top_k logits | f64 full_lse
//! ```

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::kl::{log_sum_exp, SparseRow};

const MAGIC: &[u8; 4] = b"RQLG";
const VERSION: u32 = 1;
/// Sentinel target id meaning "this position has no next token" (end of a sequence).
pub const NO_TARGET: u32 = u32::MAX;

/// One captured next-token distribution.
#[derive(Debug, Clone)]
pub enum LogitRow {
    /// Full vocabulary logits.
    Dense(Vec<f32>),
    /// Top-k logits plus the log-sum-exp of the full row, so tail mass stays recoverable.
    Sparse { ids: Vec<u32>, logits: Vec<f32>, full_lse: f64 },
}

impl LogitRow {
    /// Truncate a dense row to its top `k` entries, preserving the exact full log-sum-exp.
    pub fn sparsify(logits: &[f32], k: usize) -> LogitRow {
        let full_lse = log_sum_exp(logits);
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k.min(logits.len()));
        let vals = idx.iter().map(|&i| logits[i as usize]).collect();
        LogitRow::Sparse { ids: idx, logits: vals, full_lse }
    }

    pub fn logprob_of(&self, token: u32) -> Option<f64> {
        match self {
            LogitRow::Dense(v) => {
                let t = token as usize;
                if t >= v.len() {
                    return None;
                }
                Some((v[t] as f64) - log_sum_exp(v))
            }
            LogitRow::Sparse { ids, logits, full_lse } => ids
                .iter()
                .position(|&i| i == token)
                .map(|j| (logits[j] as f64) - full_lse),
        }
    }

    pub fn argmax(&self) -> Option<u32> {
        match self {
            LogitRow::Dense(v) => v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32),
            // Sparse rows are stored in descending order by construction.
            LogitRow::Sparse { ids, .. } => ids.first().copied(),
        }
    }
}

/// A capture of next-token distributions over some corpus.
#[derive(Debug, Clone)]
pub struct LogitStore {
    pub vocab: u32,
    /// 0 for dense rows.
    pub top_k: u32,
    /// Actual next token at each position; `NO_TARGET` where there isn't one.
    pub targets: Vec<u32>,
    pub rows: Vec<LogitRow>,
}

impl LogitStore {
    pub fn new(vocab: u32, top_k: u32) -> Self {
        Self { vocab, top_k, targets: Vec::new(), rows: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn push(&mut self, row: LogitRow, target: u32) {
        self.rows.push(row);
        self.targets.push(target);
    }

    /// Build a store from a flat `n_rows × vocab` fp32 buffer — the shape of the simplest possible
    /// dump script (`np.asarray(logits, dtype=np.float32).tofile(path)`).
    pub fn from_dense_f32(data: &[f32], vocab: usize, targets: Option<&[u32]>) -> Result<Self> {
        if vocab == 0 || data.len() % vocab != 0 {
            bail!("dense logit dump of {} floats is not a multiple of vocab {vocab}", data.len());
        }
        let n = data.len() / vocab;
        if let Some(t) = targets {
            if t.len() != n {
                bail!("{} targets for {n} rows", t.len());
            }
        }
        let mut store = Self::new(vocab as u32, 0);
        for i in 0..n {
            store.push(
                LogitRow::Dense(data[i * vocab..(i + 1) * vocab].to_vec()),
                targets.map_or(NO_TARGET, |t| t[i]),
            );
        }
        Ok(store)
    }

    /// Read a bare fp32 dump from disk (no header) and wrap it.
    pub fn read_raw_f32<P: AsRef<Path>>(path: P, vocab: usize) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("reading logit dump {}", path.as_ref().display()))?;
        if bytes.len() % 4 != 0 {
            bail!("{}: not a multiple of 4 bytes", path.as_ref().display());
        }
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::from_dense_f32(&floats, vocab, None)
    }

    /// Reduce every dense row to its top `k`, shrinking the store by ~`vocab/k`.
    pub fn sparsify(&mut self, k: usize) {
        if k == 0 {
            return;
        }
        for row in &mut self.rows {
            let replacement = match row {
                LogitRow::Dense(v) => Some(LogitRow::sparsify(v, k)),
                LogitRow::Sparse { .. } => None,
            };
            if let Some(r) = replacement {
                *row = r;
            }
        }
        self.top_k = k as u32;
    }

    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut f = std::fs::File::create(path.as_ref())
            .with_context(|| format!("creating {}", path.as_ref().display()))?;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&self.vocab.to_le_bytes());
        buf.extend_from_slice(&self.top_k.to_le_bytes());
        buf.extend_from_slice(&(self.rows.len() as u64).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // flags
        for t in &self.targets {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        for row in &self.rows {
            match row {
                LogitRow::Dense(v) => {
                    if self.top_k != 0 {
                        bail!("store declares top_k={} but holds a dense row", self.top_k);
                    }
                    for x in v {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                }
                LogitRow::Sparse { ids, logits, full_lse } => {
                    if ids.len() != self.top_k as usize {
                        bail!("store declares top_k={} but a row holds {}", self.top_k, ids.len());
                    }
                    for i in ids {
                        buf.extend_from_slice(&i.to_le_bytes());
                    }
                    for l in logits {
                        buf.extend_from_slice(&l.to_le_bytes());
                    }
                    buf.extend_from_slice(&full_lse.to_le_bytes());
                }
            }
        }
        f.write_all(&buf)
            .with_context(|| format!("writing {}", path.as_ref().display()))?;
        Ok(())
    }

    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut f = std::fs::File::open(path.as_ref())
            .with_context(|| format!("opening {}", path.as_ref().display()))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::from_bytes(&buf).with_context(|| format!("parsing {}", path.as_ref().display()))
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        // The fixed header is 4 + 4 + 4 + 4 + 8 + 8 = 32 bytes.
        if buf.len() < 32 || &buf[0..4] != MAGIC {
            bail!("not a RQLG logit store");
        }
        let u32_at = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        let version = u32_at(4);
        if version != VERSION {
            bail!("RQLG version {version} not supported (this build reads v{VERSION})");
        }
        let vocab = u32_at(8);
        let top_k = u32_at(12);
        let n_rows = u64_at(16) as usize;
        let mut cur = 32usize;

        let need = |cur: usize, n: usize, buf: &[u8]| -> Result<()> {
            if cur + n > buf.len() {
                bail!("RQLG truncated: need {n} bytes at {cur}, have {}", buf.len());
            }
            Ok(())
        };

        need(cur, n_rows * 4, buf)?;
        let mut targets = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            targets.push(u32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap()));
            cur += 4;
        }

        let mut rows = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            if top_k == 0 {
                let n = vocab as usize;
                need(cur, n * 4, buf)?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(f32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap()));
                    cur += 4;
                }
                rows.push(LogitRow::Dense(v));
            } else {
                let k = top_k as usize;
                need(cur, k * 8 + 8, buf)?;
                let mut ids = Vec::with_capacity(k);
                for _ in 0..k {
                    ids.push(u32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap()));
                    cur += 4;
                }
                let mut logits = Vec::with_capacity(k);
                for _ in 0..k {
                    logits.push(f32::from_le_bytes(buf[cur..cur + 4].try_into().unwrap()));
                    cur += 4;
                }
                let full_lse = f64::from_le_bytes(buf[cur..cur + 8].try_into().unwrap());
                cur += 8;
                rows.push(LogitRow::Sparse { ids, logits, full_lse });
            }
        }
        Ok(Self { vocab, top_k, targets, rows })
    }

    /// Compare this store (the reference) against a candidate, position by position.
    pub fn compare(&self, candidate: &LogitStore) -> Result<crate::kl::KlStats> {
        if self.vocab != candidate.vocab {
            bail!(
                "vocab mismatch: reference {} vs candidate {} — the captures are of different models",
                self.vocab,
                candidate.vocab
            );
        }
        if self.rows.len() != candidate.rows.len() {
            bail!(
                "row-count mismatch: reference {} vs candidate {} — the captures used different \
                 corpora or context lengths, so position i is not the same token",
                self.rows.len(),
                candidate.rows.len()
            );
        }
        let mut acc = crate::kl::KlAccumulator::new();
        for (i, (r, c)) in self.rows.iter().zip(&candidate.rows).enumerate() {
            let kl = match (r, c) {
                // The common case: two dense rows, compared exactly with no allocation.
                (LogitRow::Dense(a), LogitRow::Dense(b)) => crate::kl::kl_divergence(a, b),
                // Mixed or sparse: materialise both as (ids, logits, lse) triples. This copies,
                // which is why the dense/dense fast path above exists.
                _ => {
                    let (rid, rlog, rlse) = row_parts(r);
                    let (cid, clog, clse) = row_parts(c);
                    crate::kl::kl_divergence_sparse(
                        &SparseRow {
                            ids: &rid,
                            logits: &rlog,
                            full_lse: rlse,
                            vocab: self.vocab as usize,
                        },
                        &SparseRow {
                            ids: &cid,
                            logits: &clog,
                            full_lse: clse,
                            vocab: candidate.vocab as usize,
                        },
                    )
                }
            };
            let agree = match (r.argmax(), c.argmax()) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            acc.push(kl, agree);
            let t = self.targets.get(i).copied().unwrap_or(NO_TARGET);
            if t != NO_TARGET {
                if let (Some(lp), Some(lq)) = (r.logprob_of(t), c.logprob_of(t)) {
                    acc.push_target(lp, lq);
                }
            }
        }
        Ok(acc.finish())
    }
}

/// Flatten a row into `(ids, logits, full_lse)` regardless of how it was stored.
fn row_parts(row: &LogitRow) -> (Vec<u32>, Vec<f32>, f64) {
    match row {
        LogitRow::Dense(v) => ((0..v.len() as u32).collect(), v.clone(), log_sum_exp(v)),
        LogitRow::Sparse { ids, logits, full_lse } => (ids.clone(), logits.clone(), *full_lse),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(vocab: usize, n: usize) -> LogitStore {
        let mut s = LogitStore::new(vocab as u32, 0);
        for i in 0..n {
            let row: Vec<f32> = (0..vocab).map(|v| ((i * 7 + v * 3) % 11) as f32 * 0.3).collect();
            s.push(LogitRow::Dense(row), (i % vocab) as u32);
        }
        s
    }

    #[test]
    fn dense_store_round_trips_through_the_file_format() {
        let s = store(16, 5);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ref.rqlg");
        s.write_to(&p).unwrap();
        let back = LogitStore::read(&p).unwrap();
        assert_eq!(back.vocab, 16);
        assert_eq!(back.top_k, 0);
        assert_eq!(back.len(), 5);
        assert_eq!(back.targets, s.targets);
        for (a, b) in s.rows.iter().zip(&back.rows) {
            match (a, b) {
                (LogitRow::Dense(x), LogitRow::Dense(y)) => assert_eq!(x, y),
                _ => panic!("row kind changed"),
            }
        }
    }

    #[test]
    fn sparse_store_round_trips_and_keeps_the_full_lse() {
        let mut s = store(64, 4);
        let dense_lse: Vec<f64> = s
            .rows
            .iter()
            .map(|r| match r {
                LogitRow::Dense(v) => crate::kl::log_sum_exp(v),
                _ => unreachable!(),
            })
            .collect();
        s.sparsify(8);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ref.rqlg");
        s.write_to(&p).unwrap();
        let back = LogitStore::read(&p).unwrap();
        assert_eq!(back.top_k, 8);
        for (i, r) in back.rows.iter().enumerate() {
            match r {
                LogitRow::Sparse { ids, full_lse, .. } => {
                    assert_eq!(ids.len(), 8);
                    assert!((full_lse - dense_lse[i]).abs() < 1e-9);
                }
                _ => panic!("expected sparse"),
            }
        }
    }

    #[test]
    fn comparing_a_store_with_itself_is_zero_kl() {
        let s = store(32, 12);
        let stats = s.compare(&s).unwrap();
        assert_eq!(stats.n, 12);
        assert!(stats.mean < 1e-12, "{}", stats.mean);
        assert!((stats.top1_agreement - 1.0).abs() < 1e-12);
        assert!(stats.reference_ppl.is_some());
    }

    #[test]
    fn mismatched_captures_are_rejected_rather_than_silently_compared() {
        let a = store(32, 12);
        let b = store(32, 11);
        let err = a.compare(&b).unwrap_err().to_string();
        assert!(err.contains("row-count mismatch"), "{err}");

        let c = store(16, 12);
        let err = a.compare(&c).unwrap_err().to_string();
        assert!(err.contains("vocab mismatch"), "{err}");
    }

    #[test]
    fn raw_f32_import_matches_the_shape_it_was_given() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let s = LogitStore::from_dense_f32(&data, 8, None).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.vocab, 8);
        assert!(s.targets.iter().all(|&t| t == NO_TARGET));
        assert!(LogitStore::from_dense_f32(&data, 7, None).is_err());
    }

    #[test]
    fn sparsify_keeps_the_largest_logits_in_order() {
        let logits = [1.0f32, 9.0, 3.0, 7.0, 5.0];
        let row = LogitRow::sparsify(&logits, 3);
        match row {
            LogitRow::Sparse { ids, logits: v, .. } => {
                assert_eq!(ids, vec![1, 3, 4]);
                assert_eq!(v, vec![9.0, 7.0, 5.0]);
            }
            _ => panic!(),
        }
    }
}
