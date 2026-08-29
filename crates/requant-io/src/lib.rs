//! requant-io: GGUF + safetensors read/write, mmap loader, role-tagged tensor IR, and the
//! block-scaled float (FP4/FP8/MX/NVFP4) codecs that let an already-quantized checkpoint be a
//! *source*.

pub mod blockfloat;
pub mod gguf;
pub mod ir;
pub mod safetensors;

pub use blockfloat::{
    dequant_fp8, dequant_mxfp4_ggml_row, dequant_mxfp4_ocp, dequant_mxfp8, dequant_nvfp4,
    dequantize_blockfloat, e2m1_to_f32, e4m3_to_f32, e5m2_to_f32, e8m0_to_f32, e8m0_to_f32_half,
    f32_to_e2m1, f32_to_e2m1_with, f32_to_e4m3, f32_to_e5m2, f32_to_e8m0, is_fp4_family,
    is_fp8_family, is_gguf_type, E2m1Rounding, Fp8Scale, NibbleOrder, ScaleSource, E2M1_EMAX,
    E2M1_MAX, E2M1_TABLE, E4M3_EMAX, E4M3_MAX, E5M2_MAX, GGML_TYPE_MXFP4, RQ_TYPE_BASE,
    RQ_TYPE_FP8_E4M3, RQ_TYPE_FP8_E5M2, RQ_TYPE_MXFP4_OCP, RQ_TYPE_MXFP8_E4M3, RQ_TYPE_NVFP4,
};
pub use gguf::{block_layout, bpw, ggml_type_name, is_float_type, packed_nbytes, tensor_nbytes, write_gguf_streaming, GgufReader, GgufValue, GgufWriter, TensorInfo as GgufTensorInfo, TensorPlan, TensorSpec};
pub use ir::{FfnPart, MlaPart, ModelLayout, Place, Role, TensorInfo, TensorTag, TensorView};
pub use safetensors::{
    load_fp8_linear, load_nvfp4_linear, DequantizedLinear, SafeTensors, ShardedSafeTensors,
    StDtype, StEntry, TensorSource,
};
