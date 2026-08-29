//! GGUF container reader/writer (hand-written, bit-exact against the llama.cpp ecosystem).
//!
//! Layout (GGUF v3, little-endian) — confirmed from `gguf-py/gguf/gguf_writer.py`:
//!   magic "GGUF" (4 bytes) | version u32 | n_tensors u64 | n_kv u64
//!   kv*: key-string (no type prefix) | vtype u32 | value
//!   tensor-info*: name-string | n_dims u32 | dims (n_dims × u64, stored in *ne* order:
//!                 innermost/contiguous dim first, i.e. reversed from logical shape) | ggml_type u32 | offset u64
//!   padding to `general.alignment` (default 32)
//!   tensor data, each tensor placed at its (relative) offset and the file padded to alignment after each.
//!
//! A string is `u64 length` + utf-8 bytes (no terminator). An array value is `u32 elem_type | u64 n | n values`
//! (each value with no per-element type prefix). Offsets are relative to the data section start.

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const GGUF_VERSION: u32 = 3;
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF KV value types (matches `enum gguf_type`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8, 1 => Self::I8, 2 => Self::U16, 3 => Self::I16,
            4 => Self::U32, 5 => Self::I32, 6 => Self::F32, 7 => Self::Bool,
            8 => Self::String, 9 => Self::Array, 10 => Self::U64, 11 => Self::I64,
            12 => Self::F64, n => bail!("unknown gguf type {n}"),
        })
    }
}

/// A KV value. Arrays carry their element type and a homogeneous list.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array { elem: GgufType, items: Vec<GgufValue> },
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::String(s) = self { Some(s) } else { None }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            GgufValue::I32(v) => Some(*v as f32),
            GgufValue::U32(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<(&GgufType, &[GgufValue])> {
        if let GgufValue::Array { elem, items } = self { Some((elem, items)) } else { None }
    }
}

/// Tensor metadata as read from the header. `dims` is in *logical* order (outermost first),
/// i.e. for a 2-D weight `[out, in]` with `in` contiguous. `offset` is relative to the data section.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

impl TensorInfo {
    /// Number of elements (product of logical dims).
    pub fn n_elems(&self) -> u64 {
        self.dims.iter().product()
    }
    /// For a 2-D weight: (rows=out, cols=in). For 1-D: (rows=1, cols=n).
    pub fn rows_cols(&self) -> (usize, usize) {
        match self.dims.len() {
            0 => (0, 0),
            1 => (1, self.dims[0] as usize),
            _ => (self.dims[0] as usize, self.dims[1] as usize),
        }
    }
}

/// A memory-mapped GGUF reader. Tensor data is accessed zero-copy via the mmap.
pub struct GgufReader {
    #[allow(dead_code)]
    _file: File,
    mmap: Mmap,
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub kv: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorInfo>,
}

impl GgufReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("opening {}", path.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cur = 0usize;
        let magic = read_bytes(&mmap, &mut cur, 4)?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic={:x?})", magic);
        }
        let version = read_u32(&mmap, &mut cur)?;
        if version != 2 && version != 3 {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = read_u64(&mmap, &mut cur)?;
        let n_kv = read_u64(&mmap, &mut cur)?;

        let mut kv = Vec::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_string(&mmap, &mut cur)?;
            let vtype = GgufType::from_u32(read_u32(&mmap, &mut cur)?)?;
            let value = read_value(&mmap, &mut cur, vtype)?;
            kv.push((key, value));
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(&mmap, &mut cur)?;
            let n_dims = read_u32(&mmap, &mut cur)? as usize;
            let mut disk_dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                disk_dims.push(read_u64(&mmap, &mut cur)?);
            }
            let ggml_type = read_u32(&mmap, &mut cur)?;
            let offset = read_u64(&mmap, &mut cur)?;
            // disk dims are in ne order (innermost first); reverse to logical order.
            let dims: Vec<u64> = disk_dims.into_iter().rev().collect();
            tensors.push(TensorInfo { name, dims, ggml_type, offset });
        }

        let alignment = kv
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u32())
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);

        let data_offset = align_up(cur as u64, alignment);

        Ok(Self { _file: file, mmap, version, alignment, data_offset, kv, tensors })
    }

    /// Raw bytes of a tensor in the file (zero-copy view into the mmap).
    pub fn tensor_bytes(&self, idx: usize) -> Result<&[u8]> {
        let t = self.tensors.get(idx).ok_or_else(|| anyhow!("tensor idx {idx} out of range"))?;
        let start = self.data_offset as usize + t.offset as usize;
        let len = tensor_nbytes(t)? as usize;
        let end = start + len;
        if end > self.mmap.len() {
            bail!("tensor {} extends past end of file", t.name);
        }
        Ok(&self.mmap[start..end])
    }

    pub fn find_tensor(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Convert a source tensor (F32/F16/BF16) to f32. Returns the flattened logical-order data.
    pub fn tensor_to_f32(&self, idx: usize) -> Result<Vec<f32>> {
        let info = &self.tensors[idx];
        let bytes = self.tensor_bytes(idx)?;
        let n = info.n_elems() as usize;
        let mut out = Vec::with_capacity(n);
        match info.ggml_type {
            0 => {
                // F32
                for c in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            1 => {
                // F16
                for c in bytes.chunks_exact(2) {
                    out.push(f16::from_le_bytes([c[0], c[1]]).to_f32());
                }
            }
            30 => {
                // BF16
                for c in bytes.chunks_exact(2) {
                    out.push(bf16::from_le_bytes([c[0], c[1]]).to_f32());
                }
            }
            other => bail!("tensor {} has non-float source type {} (quant→quant not supported yet)", info.name, other),
        }
        if out.len() != n {
            bail!("tensor {} byte count {} doesn't match {} elements of type {}", info.name, bytes.len(), n, info.ggml_type);
        }
        Ok(out)
    }
}

/// A tensor to be written: logical-order dims `[out, in]`, ggml type, and packed data bytes.
#[derive(Clone)]
pub struct TensorSpec {
    pub name: String,
    pub dims: Vec<u64>, // logical order (outermost first); written reversed to ne order on disk
    pub ggml_type: u32,
    pub data: Vec<u8>,
}

/// A GGUF writer. KV pairs are emitted in insertion order; tensors in insertion order.
pub struct GgufWriter {
    pub kv: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorSpec>,
    pub alignment: u64,
    pub version: u32,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self { kv: Vec::new(), tensors: Vec::new(), alignment: DEFAULT_ALIGNMENT, version: GGUF_VERSION }
    }

    pub fn set_alignment(&mut self, a: u64) {
        self.alignment = a.max(1);
    }

    pub fn add_kv(&mut self, key: impl Into<String>, value: GgufValue) {
        self.kv.push((key.into(), value));
    }

    pub fn add_tensor(&mut self, spec: TensorSpec) {
        self.tensors.push(spec);
    }

    /// Copy all KV from a reader (preserving order), except `general.alignment` which we manage.
    pub fn copy_kv_from(&mut self, reader: &GgufReader) {
        for (k, v) in &reader.kv {
            if k == "general.alignment" {
                continue;
            }
            self.kv.push((k.clone(), v.clone()));
        }
    }

    /// Serialize to an in-memory buffer (the same bytes `write_to` would produce).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        // header
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());

        // KV
        for (key, val) in &self.kv {
            write_string(&mut buf, key);
            let vtype = value_type(val);
            buf.extend_from_slice(&(vtype as u32).to_le_bytes());
            write_value(&mut buf, val);
        }

        // tensor info — compute offsets as we go (relative to data section start)
        let mut offset: u64 = 0;
        for t in &self.tensors {
            write_string(&mut buf, &t.name);
            let n_dims = t.dims.len() as u32;
            buf.extend_from_slice(&n_dims.to_le_bytes());
            // disk order = ne order = reversed logical order
            for &d in t.dims.iter().rev() {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.ggml_type.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            let nbytes = t.data.len() as u64;
            offset += align_up(nbytes, self.alignment);
        }

        // pad header to alignment
        let data_offset = align_up(buf.len() as u64, self.alignment) as usize;
        let pad = data_offset - buf.len();
        buf.extend(std::iter::repeat(0u8).take(pad));

        // tensor data, each padded to alignment
        for t in &self.tensors {
            buf.extend_from_slice(&t.data);
            let p = align_up(t.data.len() as u64, self.alignment) as usize - t.data.len();
            buf.extend(std::iter::repeat(0u8).take(p));
        }
        buf
    }

    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let buf = self.to_bytes();
        std::fs::write(path.as_ref(), &buf)
            .with_context(|| format!("writing {}", path.as_ref().display()))?;
        Ok(())
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// A tensor's header entry when its data will be supplied lazily.
#[derive(Debug, Clone)]
pub struct TensorPlan {
    pub name: String,
    /// Logical order (outermost first); written reversed to ne order on disk.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Packed size — must match exactly what the provider returns.
    pub nbytes: u64,
}

/// Write a GGUF whose tensor data is produced one tensor at a time.
///
/// [`GgufWriter`] holds every tensor's bytes before serialising, which is fine for a one-shot
/// quantize but not for the sensitivity harness: that writes one whole candidate model per
/// (tensor, bits) pair, and holding a 30B model in RAM per candidate is the difference between
/// the loop running and the loop swapping. Here the header is computed from `plans` alone — sizes
/// are pure format geometry, so they're known before any data exists — and each tensor's bytes are
/// requested, written, and dropped.
///
/// `provide(i)` must return exactly `plans[i].nbytes` bytes.
pub fn write_gguf_streaming<P, F>(
    path: P,
    kv: &[(String, GgufValue)],
    plans: &[TensorPlan],
    alignment: u64,
    version: u32,
    mut provide: F,
) -> Result<u64>
where
    P: AsRef<Path>,
    F: FnMut(usize) -> Result<Vec<u8>>,
{
    use std::io::{BufWriter, Write};

    let alignment = alignment.max(1);
    let mut header: Vec<u8> = Vec::new();
    header.extend_from_slice(&GGUF_MAGIC);
    header.extend_from_slice(&version.to_le_bytes());
    header.extend_from_slice(&(plans.len() as u64).to_le_bytes());
    header.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (key, val) in kv {
        write_string(&mut header, key);
        header.extend_from_slice(&(value_type(val) as u32).to_le_bytes());
        write_value(&mut header, val);
    }
    let mut offset: u64 = 0;
    for p in plans {
        write_string(&mut header, &p.name);
        header.extend_from_slice(&(p.dims.len() as u32).to_le_bytes());
        for &d in p.dims.iter().rev() {
            header.extend_from_slice(&d.to_le_bytes());
        }
        header.extend_from_slice(&p.ggml_type.to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
        offset += align_up(p.nbytes, alignment);
    }
    let data_offset = align_up(header.len() as u64, alignment) as usize;
    header.resize(data_offset, 0);

    let file = File::create(path.as_ref())
        .with_context(|| format!("creating {}", path.as_ref().display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    w.write_all(&header)?;

    let pad = vec![0u8; alignment as usize];
    let mut total = header.len() as u64;
    for (i, p) in plans.iter().enumerate() {
        let data = provide(i)?;
        if data.len() as u64 != p.nbytes {
            bail!(
                "tensor `{}`: provider returned {} bytes, header says {}",
                p.name,
                data.len(),
                p.nbytes
            );
        }
        w.write_all(&data)?;
        let padding = align_up(p.nbytes, alignment) - p.nbytes;
        if padding > 0 {
            w.write_all(&pad[..padding as usize])?;
        }
        total += align_up(p.nbytes, alignment);
    }
    w.flush()?;
    Ok(total)
}

// ---------- helpers ----------

fn align_up(x: u64, a: u64) -> u64 {
    let a = a.max(1);
    ((x + a - 1) / a) * a
}

fn read_bytes<'a>(m: &'a [u8], cur: &mut usize, n: usize) -> Result<&'a [u8]> {
    if *cur + n > m.len() {
        bail!("gguf read past EOF at {cur}+{n}");
    }
    let s = &m[*cur..*cur + n];
    *cur += n;
    Ok(s)
}
fn read_u32(m: &[u8], cur: &mut usize) -> Result<u32> {
    Ok(u32::from_le_bytes(read_bytes(m, cur, 4)?.try_into().unwrap()))
}
fn read_u64(m: &[u8], cur: &mut usize) -> Result<u64> {
    Ok(u64::from_le_bytes(read_bytes(m, cur, 8)?.try_into().unwrap()))
}
fn read_i32(m: &[u8], cur: &mut usize) -> Result<i32> {
    Ok(i32::from_le_bytes(read_bytes(m, cur, 4)?.try_into().unwrap()))
}
fn read_i64(m: &[u8], cur: &mut usize) -> Result<i64> {
    Ok(i64::from_le_bytes(read_bytes(m, cur, 8)?.try_into().unwrap()))
}
fn read_f32(m: &[u8], cur: &mut usize) -> Result<f32> {
    Ok(f32::from_le_bytes(read_bytes(m, cur, 4)?.try_into().unwrap()))
}
fn read_f64(m: &[u8], cur: &mut usize) -> Result<f64> {
    Ok(f64::from_le_bytes(read_bytes(m, cur, 8)?.try_into().unwrap()))
}
fn read_u16(m: &[u8], cur: &mut usize) -> Result<u16> {
    Ok(u16::from_le_bytes(read_bytes(m, cur, 2)?.try_into().unwrap()))
}
fn read_i16(m: &[u8], cur: &mut usize) -> Result<i16> {
    Ok(i16::from_le_bytes(read_bytes(m, cur, 2)?.try_into().unwrap()))
}
fn read_u8(m: &[u8], cur: &mut usize) -> Result<u8> {
    Ok(read_bytes(m, cur, 1)?[0])
}
fn read_string(m: &[u8], cur: &mut usize) -> Result<String> {
    let n = read_u64(m, cur)? as usize;
    let s = std::str::from_utf8(read_bytes(m, cur, n)?)
        .with_context(|| "non-utf8 gguf string")?
        .to_string();
    Ok(s)
}

fn read_value(m: &[u8], cur: &mut usize, t: GgufType) -> Result<GgufValue> {
    Ok(match t {
        GgufType::U8 => GgufValue::U8(read_u8(m, cur)?),
        GgufType::I8 => GgufValue::I8(read_u8(m, cur)? as i8),
        GgufType::U16 => GgufValue::U16(read_u16(m, cur)?),
        GgufType::I16 => GgufValue::I16(read_i16(m, cur)?),
        GgufType::U32 => GgufValue::U32(read_u32(m, cur)?),
        GgufType::I32 => GgufValue::I32(read_i32(m, cur)?),
        GgufType::F32 => GgufValue::F32(read_f32(m, cur)?),
        GgufType::Bool => GgufValue::Bool(read_u8(m, cur)? != 0),
        GgufType::String => GgufValue::String(read_string(m, cur)?),
        GgufType::U64 => GgufValue::U64(read_u64(m, cur)?),
        GgufType::I64 => GgufValue::I64(read_i64(m, cur)?),
        GgufType::F64 => GgufValue::F64(read_f64(m, cur)?),
        GgufType::Array => {
            let elem = GgufType::from_u32(read_u32(m, cur)?)?;
            let n = read_u64(m, cur)? as usize;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                items.push(read_value(m, cur, elem)?);
            }
            GgufValue::Array { elem, items }
        }
    })
}

fn value_type(v: &GgufValue) -> GgufType {
    match v {
        GgufValue::U8(_) => GgufType::U8,
        GgufValue::I8(_) => GgufType::I8,
        GgufValue::U16(_) => GgufType::U16,
        GgufValue::I16(_) => GgufType::I16,
        GgufValue::U32(_) => GgufType::U32,
        GgufValue::I32(_) => GgufType::I32,
        GgufValue::F32(_) => GgufType::F32,
        GgufValue::Bool(_) => GgufType::Bool,
        GgufValue::String(_) => GgufType::String,
        GgufValue::U64(_) => GgufType::U64,
        GgufValue::I64(_) => GgufType::I64,
        GgufValue::F64(_) => GgufType::F64,
        GgufValue::Array { .. } => GgufType::Array,
    }
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

fn write_value(buf: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::U8(x) => buf.push(*x),
        GgufValue::I8(x) => buf.push(*x as u8),
        GgufValue::U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Bool(x) => buf.push(if *x { 1 } else { 0 }),
        GgufValue::String(s) => write_string(buf, s),
        GgufValue::U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Array { elem, items } => {
            buf.extend_from_slice(&(*elem as u32).to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                write_value(buf, it);
            }
        }
    }
}

/// Block geometry for a ggml type: `(elements_per_block, bytes_per_block)`.
/// Float types use block=1; standard quants use their GGML block; K/I/T quants use 256.
/// This is pure format geometry (the source of truth for packed sizes) — the quant crate
/// depends on it so tensor sizing stays in one place.
pub fn block_layout(ggml_type: u32) -> Option<(usize, usize)> {
    // (elements_per_block, bytes_per_block). Sizes verified against ggml-common.h block structs.
    match ggml_type {
        0  => Some((1, 4)),    // F32
        1  => Some((1, 2)),    // F16
        30 => Some((1, 2)),    // BF16
        // legacy: block = 32
        2  => Some((32, 18)),   // Q4_0: d(2) + qs[16]
        3  => Some((32, 20)),   // Q4_1: d(2) + m(2) + qs[16]
        6  => Some((32, 22)),   // Q5_0: d(2) + qh[4] + qs[16]
        7  => Some((32, 24)),   // Q5_1: d(2) + m(2) + qh[4] + qs[16]
        8  => Some((32, 34)),   // Q8_0: d(2) + qs[32]
        9  => Some((32, 36)),   // Q8_1: d(2) + m(2) + qs[32]
        41 => Some((128, 18)),  // Q1_0: d(2) + sign bits[16]
        42 => Some((64, 18)),   // Q2_0: d(2) + two-bit quants[16]
        // k-quants: super-block = 256 (QK_K). bytes per 256-elem super-block:
        10 => Some((256, 84)),   // Q2_K:  scales[16] + qs[64]   + d(2) + dmin(2)
        11 => Some((256, 110)),  // Q3_K:  hmask[32] + qs[64] + scales[12] + d(2)
        12 => Some((256, 144)),  // Q4_K:  d(2) + dmin(2) + scales[12] + qs[128]
        13 => Some((256, 176)),  // Q5_K:  d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
        14 => Some((256, 210)),  // Q6_K:  qs[128] + qh[64] + scales[16] + d(2)
        15 => Some((256, 292)),  // Q8_K:  d(f32) + qs[256] + bsums[16](i16), internal in ggml
        // i-quants (codebook family). Block = 256 (QK_K) except IQ4_NL (block 32 = QK4_NL).
        // bytes-per-256 verified against the block_iq* structs in ggml-common.h:
        16 => Some((256, 66)),   // IQ2_XXS: d(2) + qs[64] (16×uint16 codebook idx)
        17 => Some((256, 74)),   // IQ2_XS:  d(2) + qs[64] + qh[8]
        18 => Some((256, 98)),   // IQ3_XXS: d(2) + 3*(qs[32])  (scales+qs layout)
        19 => Some((256, 50)),   // IQ1_S:   d(2) + qs[32] + qh[16]
        20 => Some((32, 18)),    // IQ4_NL:  d(2) + qs[16]   (block 32 = QK4_NL)
        21 => Some((256, 110)),  // IQ3_S:   d(2) + scales[104] + qs[4]  (13*(QK_K/32) + IQ3S_N_SCALE + d)
        22 => Some((256, 82)),   // IQ2_S:   d(2) + qs[64] + scales[16]
        23 => Some((256, 136)),  // IQ4_XS:  d(2) + d2(2) + qh[4] + qs[128]
        29 => Some((256, 56)),   // IQ1_M:   qs[32] + qh[16] + scales[8]  (no super-block d)
        // Balanced ternary formats.
        34 => Some((256, 54)),   // TQ1_0: qs[48] + qh[4] + d(2)
        35 => Some((256, 66)),   // TQ2_0: qs[64] + d(2)
        // MXFP4 (ggml): block_mxfp4 { e(1) + qs[16] } = 4.25 bpw.
        crate::blockfloat::GGML_TYPE_MXFP4 => Some((32, 17)),
        crate::blockfloat::GGML_TYPE_NVFP4 => Some((64, 36)),
        // ---- requant-internal block-float ids (no ggml type; see blockfloat.rs) ----
        // These report their *fully loaded* cost, i.e. element bytes plus the sidecar scale
        // bytes, because that is the number the byte-budget search must spend. The bytes are not
        // contiguous on disk: NVFP4's scale byte lives in a sibling `weight_scale` tensor.
        crate::blockfloat::RQ_TYPE_NVFP4 => Some((16, 9)),        // 16×E2M1 (8B) + E4M3 scale (1B) = 4.5 bpw
        crate::blockfloat::RQ_TYPE_MXFP4_OCP => Some((32, 17)),   // 32×E2M1 (16B) + E8M0 scale (1B) = 4.25 bpw
        crate::blockfloat::RQ_TYPE_MXFP8_E4M3 => Some((32, 33)),  // 32×E4M3 (32B) + E8M0 scale (1B) = 8.25 bpw
        // Dense FP8: the scale tensor is per-tensor/per-channel/per-128×128-tile, i.e. a rounding
        // error on the total, so the per-element cost is exactly one byte.
        crate::blockfloat::RQ_TYPE_FP8_E4M3 => Some((1, 1)),
        crate::blockfloat::RQ_TYPE_FP8_E5M2 => Some((1, 1)),
        _ => None,
    }
}

/// Number of bytes a tensor occupies on disk for a given ggml type and element count.
pub fn tensor_nbytes(info: &TensorInfo) -> Result<u64> {
    packed_nbytes(info.ggml_type, info.n_elems(), &info.name)
}

/// Packed byte count for `n` elements of `ggml_type`. Requires `n` divisible by the block size.
pub fn packed_nbytes(ggml_type: u32, n: u64, name: &str) -> Result<u64> {
    let (block, bytes) = block_layout(ggml_type)
        .ok_or_else(|| anyhow!("unsupported ggml type {ggml_type} (tensor {name})"))?;
    if block == 1 {
        return Ok(n * bytes as u64);
    }
    if n % block as u64 != 0 {
        bail!("type {ggml_type} (tensor {name}): element count {n} not divisible by block {block}");
    }
    Ok(n / block as u64 * bytes as u64)
}

/// Bits-per-weight for a format (informational).
pub fn bpw(ggml_type: u32) -> Option<f64> {
    let (e, b) = block_layout(ggml_type)?;
    Some(b as f64 * 8.0 / e as f64)
}

/// Human-readable name for a ggml type id (inverse of `Bits::to_ggml_type`). Covers the types
/// `block_layout` knows plus the float types; unknown ids format as `type{n}`.
pub fn ggml_type_name(ggml_type: u32) -> String {
    match ggml_type {
        0 => "F32".into(),
        1 => "F16".into(),
        30 => "BF16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        6 => "Q5_0".into(),
        7 => "Q5_1".into(),
        8 => "Q8_0".into(),
        9 => "Q8_1".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        15 => "Q8_K".into(),
        16 => "IQ2_XXS".into(),
        17 => "IQ2_XS".into(),
        18 => "IQ3_XXS".into(),
        19 => "IQ1_S".into(),
        20 => "IQ4_NL".into(),
        21 => "IQ3_S".into(),
        22 => "IQ2_S".into(),
        23 => "IQ4_XS".into(),
        29 => "IQ1_M".into(),
        34 => "TQ1_0".into(),
        35 => "TQ2_0".into(),
        crate::blockfloat::GGML_TYPE_MXFP4 => "MXFP4".into(),
        crate::blockfloat::GGML_TYPE_NVFP4 => "NVFP4_GGUF".into(),
        41 => "Q1_0".into(),
        42 => "Q2_0".into(),
        crate::blockfloat::RQ_TYPE_NVFP4 => "NVFP4".into(),
        crate::blockfloat::RQ_TYPE_MXFP4_OCP => "MXFP4_OCP".into(),
        crate::blockfloat::RQ_TYPE_MXFP8_E4M3 => "MXFP8".into(),
        crate::blockfloat::RQ_TYPE_FP8_E4M3 => "FP8_E4M3".into(),
        crate::blockfloat::RQ_TYPE_FP8_E5M2 => "FP8_E5M2".into(),
        n => format!("type{n}"),
    }
}

/// True for F32/F16/BF16 — types `tensor_to_f32` can read directly.
pub fn is_float_type(ggml_type: u32) -> bool {
    matches!(ggml_type, 0 | 1 | 30)
}
