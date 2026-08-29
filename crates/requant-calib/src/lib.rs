//! requant-calib: importance-matrix (imatrix) import, native compute, content-addressed cache.
//!
//! v1 ships the imatrix file loader (`imatrix`). Native forward-pass computation of the
//! imatrix and the on-disk content-addressed cache (`cache`) land in a later phase.

pub mod imatrix;
pub mod cache;

pub use imatrix::{load_imatrix, count_nonfinite, Imatrix, ImatrixEntry};
