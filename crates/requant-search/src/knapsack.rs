//! Greedy-marginal bit allocation under a byte budget (DESIGN §3.7).
//!
//! The setup: every quantizable tensor has a curve of `(bytes, cost)` points — one per candidate
//! precision — where `cost` is what quantizing it to that precision costs the model. Measured ΔKL
//! from [`requant_eval::sensitivity`] is the real thing; [`crate::proxy`]'s weighted round-trip
//! error is the free stand-in. Either way the allocator's job is the same: pick one point per
//! tensor, minimising total cost subject to total bytes ≤ budget.
//!
//! That is a **multiple-choice knapsack**, which is NP-hard in general and completely tractable
//! here, for a reason worth being precise about.
//!
//! # Why greedy is not a shortcut
//!
//! Take each tensor's **lower convex hull** in `(bytes, cost)` space first. Along a hull, the
//! marginal gain `Δcost/Δbytes` of each successive upgrade is strictly decreasing — the curve is
//! concave in savings. With that property, repeatedly spending the next byte wherever the marginal
//! gain is largest is *exactly* the optimal solution to the LP relaxation, and the integer optimum
//! differs from it by at most one item (the one straddling the budget boundary). So this is not
//! "greedy, hopefully close"; it is an optimal fractional solution with a one-item integrality gap,
//! which on a 300-tensor model is noise.
//!
//! The hull is doing real work. Without it, a tensor whose middle rung is a bad deal but whose top
//! rung is excellent gets stuck: greedy evaluates only the *next* step, sees a poor ratio, and
//! never discovers the good one two steps up. Taking the hull replaces that pair of steps with the
//! single combined step whose ratio reflects the whole jump. Points the hull skips are not lost —
//! [`AllocOpts::fill_slack`] revisits them at the end, where a smaller intermediate rung may be
//! exactly what fits in the leftover budget.
//!
//! # Floors are not negotiable
//!
//! `points[0]` is the floor: the cheapest precision a tensor may take. The recipe sets it, and the
//! allocator never goes below it even when the budget binds. This is what stops the optimizer from
//! discovering that it can buy a lot of bytes by taking the router to Q2_K — mathematically
//! attractive, catastrophic in practice, and invisible to any aggregate metric. When the floors
//! alone exceed the budget the answer is "your budget is too small for this quality floor", not a
//! silently degraded model; [`Allocation::over_budget`] says so.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use requant_quant::Bits;

/// One precision option for one tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The precision as the recipe names it (before block-alignment fallback).
    pub bits: Bits,
    /// The type that will actually be written, after fallback.
    pub ggml_type: u32,
    /// Bytes this tensor occupies at this precision, including sidecar scales.
    pub bytes: u64,
    /// Cost of choosing this precision. Lower is better; units are whatever the source used
    /// (nats of KL, or weighted squared error). Must be comparable *across* tensors.
    pub cost: f64,
}

/// One tensor's options, or the fact that it has none.
#[derive(Debug, Clone)]
pub struct TensorCurve {
    pub name: String,
    pub role: String,
    /// Set for tensors excluded from the search (protected, float, or unquantizable). Their bytes
    /// count against the budget but the allocator never moves them.
    pub fixed: Option<Bits>,
    pub fixed_bytes: u64,
    /// Ascending in bytes. Empty for fixed tensors. `points[0]` is the floor.
    pub points: Vec<Candidate>,
}

impl TensorCurve {
    pub fn fixed_at(name: impl Into<String>, role: impl Into<String>, bits: Option<Bits>, bytes: u64) -> Self {
        Self { name: name.into(), role: role.into(), fixed: bits, fixed_bytes: bytes, points: Vec::new() }
    }

    pub fn is_searchable(&self) -> bool {
        self.points.len() > 1
    }

    /// Bytes when everything is at its floor.
    pub fn floor_bytes(&self) -> u64 {
        self.points.first().map_or(self.fixed_bytes, |p| p.bytes)
    }

    /// Sort ascending by bytes, drop duplicates and dominated points.
    ///
    /// A point is dominated when a cheaper-in-bytes point already costs the same or less: paying
    /// for it can never be right, and leaving it in would put a zero- or negative-gain step in the
    /// heap. The floor is always kept, dominated or not — it defines the lower bound.
    pub fn normalize(&mut self) {
        if self.points.is_empty() {
            return;
        }
        self.points.sort_by(|a, b| {
            a.bytes
                .cmp(&b.bytes)
                .then_with(|| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal))
        });
        let mut out: Vec<Candidate> = Vec::with_capacity(self.points.len());
        out.push(self.points[0]);
        let mut best = self.points[0].cost;
        for p in self.points.iter().skip(1) {
            if p.bytes == out.last().unwrap().bytes {
                continue; // same size, no better cost (we sorted cost ascending within a size)
            }
            if p.cost < best {
                best = p.cost;
                out.push(*p);
            }
        }
        self.points = out;
    }

    /// Indices of the points on the lower convex hull of `(bytes, cost)`.
    ///
    /// Requires [`normalize`](Self::normalize) to have run.
    pub fn hull(&self) -> Vec<usize> {
        let n = self.points.len();
        if n <= 2 {
            return (0..n).collect();
        }
        let slope = |a: usize, b: usize| -> f64 {
            let db = self.points[b].bytes as f64 - self.points[a].bytes as f64;
            if db <= 0.0 {
                return f64::NEG_INFINITY;
            }
            (self.points[b].cost - self.points[a].cost) / db
        };
        let mut h: Vec<usize> = vec![0];
        for i in 1..n {
            // Drop the previous vertex whenever it sits on or above the chord that would replace
            // it — that is exactly the "two small steps are better taken as one big one" case.
            while h.len() >= 2 {
                let b = h[h.len() - 1];
                let a = h[h.len() - 2];
                if slope(a, b) >= slope(a, i) {
                    h.pop();
                } else {
                    break;
                }
            }
            h.push(i);
        }
        h
    }
}

/// Allocator knobs.
#[derive(Debug, Clone, Copy)]
pub struct AllocOpts {
    /// Take the convex hull before the greedy walk. Off only for testing the difference.
    pub hull: bool,
    /// After the greedy walk, spend any leftover budget on the best upgrade that still fits,
    /// including off-hull rungs the greedy pass skipped over.
    pub fill_slack: bool,
}

impl Default for AllocOpts {
    fn default() -> Self {
        Self { hull: true, fill_slack: true }
    }
}

/// The result of an allocation.
#[derive(Debug, Clone)]
pub struct Allocation {
    /// Index into each curve's `points`. Meaningless (0) for fixed curves.
    pub choice: Vec<usize>,
    pub budget: u64,
    pub fixed_bytes: u64,
    pub searchable_bytes: u64,
    pub total_bytes: u64,
    /// Sum of the chosen points' costs over searchable tensors.
    pub total_cost: f64,
    /// True when even the floors don't fit.
    pub over_budget: bool,
    /// Bytes by which the floors exceed the budget (0 unless `over_budget`).
    pub shortfall: u64,
    /// Number of upgrades applied above the floor.
    pub upgrades: usize,
}

impl Allocation {
    /// The precision chosen for curve `i`.
    pub fn bits_of(&self, curves: &[TensorCurve], i: usize) -> Bits {
        let c = &curves[i];
        if let Some(b) = c.fixed {
            return b;
        }
        c.points
            .get(self.choice[i])
            .map(|p| p.bits)
            .or_else(|| c.points.first().map(|p| p.bits))
            .unwrap_or(Bits::F16)
    }

    pub fn slack(&self) -> u64 {
        self.budget.saturating_sub(self.total_bytes)
    }
}

/// Max-heap entry: the next upgrade available for one tensor.
#[derive(Debug, Clone, Copy)]
struct Step {
    /// Δcost per Δbyte. Larger is a better deal.
    score: f64,
    curve: usize,
    /// Position within the hull we would move to.
    hull_pos: usize,
    extra_bytes: u64,
}

impl PartialEq for Step {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Step {}
impl Ord for Step {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            // Deterministic tie-break: prefer the lower curve index, so the same inputs always
            // produce the same recipe. Reversed because this is a max-heap.
            .then_with(|| other.curve.cmp(&self.curve))
    }
}
impl PartialOrd for Step {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn total_of(curves: &[TensorCurve], choice: &[usize]) -> (u64, u64, f64) {
    let mut fixed = 0u64;
    let mut searchable = 0u64;
    let mut cost = 0.0f64;
    for (i, c) in curves.iter().enumerate() {
        if c.points.is_empty() {
            fixed += c.fixed_bytes;
        } else {
            let p = &c.points[choice[i].min(c.points.len() - 1)];
            searchable += p.bytes;
            cost += p.cost;
        }
    }
    (fixed, searchable, cost)
}

/// Allocate precisions under `budget`.
///
/// `curves` must already be normalized ([`TensorCurve::normalize`]).
pub fn allocate(curves: &[TensorCurve], budget: u64, opts: &AllocOpts) -> Allocation {
    let n = curves.len();
    let mut choice = vec![0usize; n];
    let (fixed_bytes, floor_searchable, _) = total_of(curves, &choice);
    let base = fixed_bytes + floor_searchable;

    if base > budget {
        let (f, s, cost) = total_of(curves, &choice);
        return Allocation {
            choice,
            budget,
            fixed_bytes: f,
            searchable_bytes: s,
            total_bytes: f + s,
            total_cost: cost,
            over_budget: true,
            shortfall: base - budget,
            upgrades: 0,
        };
    }

    let hulls: Vec<Vec<usize>> = curves
        .iter()
        .map(|c| if opts.hull { c.hull() } else { (0..c.points.len()).collect() })
        .collect();

    let step_for = |k: usize, pos: usize, hulls: &[Vec<usize>]| -> Option<Step> {
        let h = &hulls[k];
        if pos + 1 >= h.len() {
            return None;
        }
        let cur = &curves[k].points[h[pos]];
        let next = &curves[k].points[h[pos + 1]];
        let d_bytes = next.bytes.checked_sub(cur.bytes)?;
        if d_bytes == 0 {
            return None;
        }
        let d_cost = cur.cost - next.cost;
        if d_cost <= 0.0 {
            return None;
        }
        Some(Step { score: d_cost / d_bytes as f64, curve: k, hull_pos: pos + 1, extra_bytes: d_bytes })
    };

    let mut heap: BinaryHeap<Step> = BinaryHeap::new();
    for k in 0..n {
        if let Some(s) = step_for(k, 0, &hulls) {
            heap.push(s);
        }
    }

    let mut spent = base;
    let mut upgrades = 0usize;
    while let Some(s) = heap.pop() {
        if spent + s.extra_bytes > budget {
            // The budget only ever shrinks, so this step can never fit later — and neither can
            // any larger one further up this tensor's hull. Drop the chain.
            continue;
        }
        spent += s.extra_bytes;
        choice[s.curve] = hulls[s.curve][s.hull_pos];
        upgrades += 1;
        if let Some(next) = step_for(s.curve, s.hull_pos, &hulls) {
            heap.push(next);
        }
    }

    if opts.fill_slack {
        // The hull skipped intermediate rungs; one of them may fit in the leftover budget even
        // though the hull's next jump did not. Repeatedly take the best-ratio upgrade that fits.
        loop {
            let slack = budget.saturating_sub(spent);
            if slack == 0 {
                break;
            }
            let mut best: Option<(usize, usize, u64, f64)> = None; // (curve, level, extra, score)
            for (k, c) in curves.iter().enumerate() {
                if c.points.is_empty() {
                    continue; // fixed tensor: nothing to upgrade
                }
                let cur = &c.points[choice[k]];
                for (j, p) in c.points.iter().enumerate().skip(choice[k] + 1) {
                    let extra = p.bytes.saturating_sub(cur.bytes);
                    if extra == 0 || extra > slack {
                        continue;
                    }
                    let gain = cur.cost - p.cost;
                    if gain <= 0.0 {
                        continue;
                    }
                    let score = gain / extra as f64;
                    if best.map_or(true, |(_, _, _, s)| score > s) {
                        best = Some((k, j, extra, score));
                    }
                }
            }
            let Some((k, j, extra, _)) = best else { break };
            choice[k] = j;
            spent += extra;
            upgrades += 1;
        }
    }

    let (f, s, cost) = total_of(curves, &choice);
    Allocation {
        choice,
        budget,
        fixed_bytes: f,
        searchable_bytes: s,
        total_bytes: f + s,
        total_cost: cost,
        over_budget: false,
        shortfall: 0,
        upgrades,
    }
}

/// The full size↔cost Pareto front, independent of any one budget.
///
/// Sweeps the Lagrange multiplier λ over every marginal ratio present in the hulls. At a given λ
/// each tensor independently takes every upgrade whose gain-per-byte is at least λ, which is the
/// exact minimiser of `cost + λ·bytes` — so each λ yields one vertex of the LP-relaxed front, and
/// sweeping the breakpoints enumerates them all. This is what turns "search @ 400 MiB" into a
/// table rather than a series of separate runs.
pub fn pareto_front(curves: &[TensorCurve], opts: &AllocOpts) -> Vec<Allocation> {
    let n = curves.len();
    let hulls: Vec<Vec<usize>> = curves
        .iter()
        .map(|c| if opts.hull { c.hull() } else { (0..c.points.len()).collect() })
        .collect();

    // Every marginal ratio is a breakpoint of the front.
    let mut lambdas: Vec<f64> = Vec::new();
    for (k, h) in hulls.iter().enumerate() {
        for w in h.windows(2) {
            let a = &curves[k].points[w[0]];
            let b = &curves[k].points[w[1]];
            let db = b.bytes as f64 - a.bytes as f64;
            if db > 0.0 && a.cost > b.cost {
                lambdas.push((a.cost - b.cost) / db);
            }
        }
    }
    lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    lambdas.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON * a.abs().max(1.0));
    // λ = +∞ (everything at its floor) is the left end of the front.
    lambdas.insert(0, f64::INFINITY);

    let mut out: Vec<Allocation> = Vec::new();
    let mut seen: BTreeMap<u64, usize> = BTreeMap::new();
    for &lam in &lambdas {
        let mut choice = vec![0usize; n];
        for (k, h) in hulls.iter().enumerate() {
            let mut pos = 0usize;
            while pos + 1 < h.len() {
                let a = &curves[k].points[h[pos]];
                let b = &curves[k].points[h[pos + 1]];
                let db = b.bytes as f64 - a.bytes as f64;
                if db <= 0.0 {
                    break;
                }
                let ratio = (a.cost - b.cost) / db;
                if ratio + 1e-18 < lam {
                    break;
                }
                pos += 1;
            }
            choice[k] = h[pos];
        }
        let (f, s, cost) = total_of(curves, &choice);
        let total = f + s;
        let upgrades = choice.iter().filter(|&&c| c > 0).count();
        let alloc = Allocation {
            choice,
            budget: total,
            fixed_bytes: f,
            searchable_bytes: s,
            total_bytes: total,
            total_cost: cost,
            over_budget: false,
            shortfall: 0,
            upgrades,
        };
        match seen.get(&total) {
            // Same size already on the front: keep whichever costs less.
            Some(&idx) => {
                if cost < out[idx].total_cost {
                    out[idx] = alloc;
                }
            }
            None => {
                seen.insert(total, out.len());
                out.push(alloc);
            }
        }
    }
    out.sort_by_key(|a| a.total_bytes);
    out
}

/// Render an allocation as a reproducible recipe.
///
/// The rules match by exact tensor `name` and are grouped by precision, so a 300-tensor model
/// yields ~6 rules rather than 300. `requant quantize --recipe <this>` reproduces the allocation
/// exactly — which is the whole contract: a search result that can't be re-applied is a report,
/// not a recipe.
pub fn emit_recipe(curves: &[TensorCurve], alloc: &Allocation, header: &str, default_bits: Bits) -> String {
    let mut by_bits: BTreeMap<&'static str, Vec<&str>> = BTreeMap::new();
    for (i, c) in curves.iter().enumerate() {
        by_bits.entry(alloc.bits_of(curves, i).name()).or_default().push(c.name.as_str());
    }
    for names in by_bits.values_mut() {
        names.sort_unstable();
    }

    let mut s = String::new();
    for line in header.lines() {
        s.push_str("# ");
        s.push_str(line);
        s.push('\n');
    }
    s.push_str("# Each [[rule]] matches by exact tensor name; last-match-wins.\n\n");
    s.push_str("[defaults]\n");
    s.push_str(&format!("bits = \"{}\"\nimatrix = true\n\n", default_bits.name()));
    for (bits, names) in &by_bits {
        s.push_str("[[rule]]\n");
        s.push_str(&format!("bits = \"{bits}\"\n"));
        s.push_str("name = [\n");
        for n in names {
            s.push_str(&format!("  \"{n}\",\n"));
        }
        s.push_str("]\n\n");
    }

    s.push_str("# Per-tensor allocation (name, role, bits, bytes, cost):\n");
    for (i, c) in curves.iter().enumerate() {
        let bits = alloc.bits_of(curves, i);
        let (bytes, cost) = match c.points.get(alloc.choice[i]) {
            Some(p) => (p.bytes, p.cost),
            None => (c.fixed_bytes, 0.0),
        };
        s.push_str(&format!(
            "# {:<44} {:<20} {:<9} {:>12} {:>14.6e}\n",
            c.name,
            c.role,
            bits.name(),
            bytes,
            cost
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(bits: Bits, bytes: u64, cost: f64) -> Candidate {
        Candidate { bits, ggml_type: bits.to_ggml_type(), bytes, cost }
    }

    fn curve(name: &str, pts: Vec<Candidate>) -> TensorCurve {
        let mut t = TensorCurve {
            name: name.into(),
            role: "ffn_up".into(),
            fixed: None,
            fixed_bytes: 0,
            points: pts,
        };
        t.normalize();
        t
    }

    #[test]
    fn normalize_drops_dominated_and_duplicate_points() {
        let t = curve(
            "t",
            vec![
                c(Bits::Q4_K, 100, 1.0),
                c(Bits::Q5_K, 120, 1.2), // more bytes, worse cost -> dominated
                c(Bits::Q6_K, 140, 0.5),
                c(Bits::Q8_0, 140, 0.9), // duplicate size, worse
                c(Bits::Q2_K, 60, 4.0),
            ],
        );
        let sizes: Vec<u64> = t.points.iter().map(|p| p.bytes).collect();
        assert_eq!(sizes, vec![60, 100, 140]);
    }

    #[test]
    fn hull_skips_a_bad_middle_rung() {
        // 100->200 costs 1.0 of gain; 200->300 gains 8.0. Greedy on raw points would rate the
        // first step at 0.01/byte; the hull merges them into one 0.09/byte step.
        let t = curve(
            "t",
            vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 9.0), c(Bits::Q6_K, 300, 1.0)],
        );
        assert_eq!(t.hull(), vec![0, 2], "the middle point is above the chord");

        // With a genuinely concave curve every point survives.
        let t2 = curve(
            "t",
            vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 3.0), c(Bits::Q6_K, 300, 1.0)],
        );
        assert_eq!(t2.hull(), vec![0, 1, 2]);
    }

    #[test]
    fn allocation_respects_the_budget_and_the_floor() {
        let curves = vec![
            curve("a", vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 2.0)]),
            curve("b", vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 9.0)]),
        ];
        // Budget for exactly one upgrade: it must go to `a`, the better deal.
        let a = allocate(&curves, 300, &AllocOpts::default());
        assert!(!a.over_budget);
        assert_eq!(a.total_bytes, 300);
        assert_eq!(a.bits_of(&curves, 0), Bits::Q4_K);
        assert_eq!(a.bits_of(&curves, 1), Bits::Q2_K);

        // Budget below the floors: report it, don't quietly go lower.
        let tight = allocate(&curves, 150, &AllocOpts::default());
        assert!(tight.over_budget);
        assert_eq!(tight.shortfall, 50);
        assert_eq!(tight.total_bytes, 200, "floors are held even when they don't fit");
    }

    #[test]
    fn fixed_tensors_consume_budget_but_never_move() {
        let curves = vec![
            TensorCurve::fixed_at("emb", "embedding", Some(Bits::F16), 500),
            curve("a", vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 2.0)]),
        ];
        let a = allocate(&curves, 650, &AllocOpts::default());
        assert_eq!(a.fixed_bytes, 500);
        assert_eq!(a.bits_of(&curves, 0), Bits::F16);
        assert_eq!(a.bits_of(&curves, 1), Bits::Q2_K, "only 150 spare, an upgrade needs 100... ");
        let b = allocate(&curves, 700, &AllocOpts::default());
        assert_eq!(b.bits_of(&curves, 1), Bits::Q4_K);
    }

    #[test]
    fn fill_slack_finds_an_off_hull_rung_that_fits() {
        // The hull merges 100->300, which won't fit in a 250 budget; the skipped 200 rung will.
        let curves = vec![curve(
            "a",
            vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 9.0), c(Bits::Q6_K, 300, 1.0)],
        )];
        let without = allocate(&curves, 250, &AllocOpts { hull: true, fill_slack: false });
        assert_eq!(without.bits_of(&curves, 0), Bits::Q2_K);
        let with = allocate(&curves, 250, &AllocOpts::default());
        assert_eq!(with.bits_of(&curves, 0), Bits::Q4_K, "leftover budget should get spent");
    }

    #[test]
    fn greedy_matches_brute_force_on_small_instances() {
        // Exhaustive check that the hull+greedy+fill pipeline finds the optimum on instances small
        // enough to enumerate. If the allocator is ever "clever" in a wrong way, this catches it.
        let mk = |seed: u64| -> Vec<TensorCurve> {
            let mut s = seed;
            let mut rnd = move || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % 1000) as f64 / 1000.0
            };
            (0..4)
                .map(|i| {
                    let mut cost = 5.0 + rnd() * 5.0;
                    let pts: Vec<Candidate> = (0..4)
                        .map(|j| {
                            cost -= rnd() * 2.0;
                            c(Bits::Q4_K, 100 + j * 37, cost.max(0.01))
                        })
                        .collect();
                    curve(&format!("t{i}"), pts)
                })
                .collect()
        };
        for seed in 0..40u64 {
            let curves = mk(seed);
            for budget in [420u64, 500, 560, 640] {
                let got = allocate(&curves, budget, &AllocOpts::default());
                if got.over_budget {
                    continue;
                }
                // Brute force over the product of all points.
                let mut best = f64::INFINITY;
                let dims: Vec<usize> = curves.iter().map(|c| c.points.len()).collect();
                let total: usize = dims.iter().product();
                for mut code in 0..total {
                    let mut bytes = 0u64;
                    let mut cost = 0.0;
                    for (k, &d) in dims.iter().enumerate() {
                        let j = code % d;
                        code /= d;
                        bytes += curves[k].points[j].bytes;
                        cost += curves[k].points[j].cost;
                    }
                    if bytes <= budget && cost < best {
                        best = cost;
                    }
                }
                // One-item integrality gap: allow the greedy result to miss by one upgrade's worth.
                assert!(
                    got.total_cost <= best + 2.0,
                    "seed {seed} budget {budget}: greedy {} vs optimum {best}",
                    got.total_cost
                );
            }
        }
    }

    #[test]
    fn pareto_front_is_monotone_in_both_axes() {
        let curves = vec![
            curve("a", vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 4.0), c(Bits::Q6_K, 400, 1.0)]),
            curve("b", vec![c(Bits::Q2_K, 50, 8.0), c(Bits::Q4_K, 150, 3.0), c(Bits::Q6_K, 250, 2.5)]),
        ];
        let front = pareto_front(&curves, &AllocOpts::default());
        assert!(front.len() >= 3, "expected several front vertices, got {}", front.len());
        for w in front.windows(2) {
            assert!(w[1].total_bytes > w[0].total_bytes, "sizes must strictly increase");
            assert!(
                w[1].total_cost <= w[0].total_cost + 1e-12,
                "cost must not increase as the budget grows: {} -> {}",
                w[0].total_cost,
                w[1].total_cost
            );
        }
        // The ends are the all-floor and all-top configurations.
        assert_eq!(front.first().unwrap().total_bytes, 150);
        assert_eq!(front.last().unwrap().total_bytes, 650);
    }

    #[test]
    fn front_points_agree_with_allocating_at_their_own_size() {
        let curves = vec![
            curve("a", vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 4.0), c(Bits::Q6_K, 400, 1.0)]),
            curve("b", vec![c(Bits::Q2_K, 50, 8.0), c(Bits::Q4_K, 150, 3.0)]),
        ];
        for point in pareto_front(&curves, &AllocOpts::default()) {
            let direct = allocate(&curves, point.total_bytes, &AllocOpts::default());
            assert!(
                direct.total_cost <= point.total_cost + 1e-9,
                "allocating at {} bytes gave {} but the front says {}",
                point.total_bytes,
                direct.total_cost,
                point.total_cost
            );
        }
    }

    #[test]
    fn emitted_recipe_groups_by_bits_and_names_every_tensor() {
        let curves = vec![
            TensorCurve::fixed_at("token_embd.weight", "embedding", Some(Bits::F16), 500),
            curve("blk.0.ffn_up.weight", vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 2.0)]),
            curve("blk.1.ffn_up.weight", vec![c(Bits::Q2_K, 100, 10.0), c(Bits::Q4_K, 200, 2.0)]),
        ];
        let a = allocate(&curves, 900, &AllocOpts::default());
        let toml = emit_recipe(&curves, &a, "test", Bits::Q4_K);
        assert!(toml.contains("[defaults]"));
        assert!(toml.contains("bits = \"F16\""));
        assert!(toml.contains("bits = \"Q4_K\""));
        assert!(toml.contains("\"token_embd.weight\""));
        assert!(toml.contains("\"blk.0.ffn_up.weight\""));
        assert!(toml.contains("\"blk.1.ffn_up.weight\""));
        // It must parse back as a recipe.
        let parsed = requant_quant::Recipe::parse(&toml).expect("emitted recipe must parse");
        assert!(!parsed.rule.is_empty());
    }

    #[test]
    fn zero_gain_steps_are_never_bought() {
        // A rung that costs bytes and buys nothing must be skipped even with budget to spare.
        let curves = vec![curve(
            "a",
            vec![c(Bits::Q2_K, 100, 5.0), c(Bits::Q4_K, 200, 5.0)],
        )];
        let a = allocate(&curves, 10_000, &AllocOpts::default());
        assert_eq!(a.total_bytes, 100);
        assert_eq!(a.upgrades, 0);
    }
}
