//! Quantizer trait + shared types.

use anyhow::Result;

/// Which calibration statistic a method needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatKind {
    /// No calibration (plain RTN).
    None,
    /// Diagonal of the Hessian = imatrix (per-input-channel importance).
    Diag,
    /// Full Gram matrix XX^T (GPTQ). Out of v1 scope.
    Gram,
    /// Per-channel activation statistics (AWQ/SmoothQuant). Out of v1 scope.
    ActScale,
}

/// A quantized tensor: packed bytes + target ggml type + shape.
#[derive(Debug, Clone)]
pub struct QuantTensor {
    pub ggml_type: u32,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

pub trait Quantizer {
    fn required_stat(&self) -> StatKind;
    fn quantize(&self, w: &[f32], shape: &[u64], importance: Option<&[f32]>) -> Result<QuantTensor>;
    fn dequantize(&self, q: &QuantTensor) -> Result<Vec<f32>>;
}
