#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_int};

pub const CblasRowMajor: CBLAS_ORDER = 101;
pub const CblasColMajor: CBLAS_ORDER = 102;
pub type CBLAS_ORDER = c_int;
pub const CblasNoTrans: CBLAS_TRANSPOSE = 111;
pub const CblasTrans: CBLAS_TRANSPOSE = 112;
pub const CblasConjTrans: CBLAS_TRANSPOSE = 113;
pub type CBLAS_TRANSPOSE = c_int;
pub const CblasUpper: CBLAS_UPLO = 121;
pub const CblasLower: CBLAS_UPLO = 122;
pub type CBLAS_UPLO = c_int;
pub const CblasNonUnit: CBLAS_DIAG = 131;
pub const CblasUnit: CBLAS_DIAG = 132;
pub type CBLAS_DIAG = c_int;
pub const CblasLeft: CBLAS_SIDE = 141;
pub const CblasRight: CBLAS_SIDE = 142;
pub type CBLAS_SIDE = c_int;

extern "C" {
  pub fn cblas_sasum(
      N: c_int,
      X: *const f32, incX: c_int)
      -> f32;
  pub fn cblas_saxpy(
      N: c_int,
      alpha: f32,
      X: *const f32, incX: c_int,
      Y: *mut f32, incY: c_int);
  pub fn cblas_scopy(
      N: c_int,
      X: *const f32, incX: c_int,
      Y: *mut f32, incY: c_int);
  pub fn cblas_sdot(
      N: c_int,
      X: *const f32, incX: c_int,
      Y: *const f32, incY: c_int)
      -> f32;
  pub fn cblas_snrm2(
      N: c_int,
      X: *const f32, incX: c_int)
      -> f32;
  pub fn cblas_sscal(
      N: c_int,
      alpha: f32,
      X: *mut f32, incX: c_int);
  pub fn cblas_sgemv(
      Order: CBLAS_ORDER,
      TransA: CBLAS_TRANSPOSE,
      M: c_int, N: c_int,
      alpha: f32,
      A: *const f32, lda: c_int,
      X: *const f32, incX: c_int,
      beta: f32,
      Y: *mut f32, incY: c_int);
  pub fn cblas_sgemm(
      Order: CBLAS_ORDER,
      TransA: CBLAS_TRANSPOSE,
      TransB: CBLAS_TRANSPOSE,
      M: c_int, N: c_int, K: c_int,
      alpha: f32,
      A: *const f32, lda: c_int,
      B: *const f32, ldb: c_int,
      beta: f32,
      C: *mut f32, ldc: c_int);
  pub fn cblas_dasum(
      N: c_int,
      X: *const f64, incX: c_int)
      -> f64;
  pub fn cblas_daxpy(
      N: c_int,
      alpha: f64,
      X: *const f64, incX: c_int,
      Y: *mut f64, incY: c_int);
  pub fn cblas_dcopy(
      N: c_int,
      X: *const f64, incX: c_int,
      Y: *mut f64, incY: c_int);
  pub fn cblas_ddot(
      N: c_int,
      X: *const f64, incX: c_int,
      Y: *const f64, incY: c_int)
      -> f64;
  pub fn cblas_dnrm2(
      N: c_int,
      X: *const f64, incX: c_int)
      -> f64;
  pub fn cblas_dscal(
      N: c_int,
      alpha: f64,
      X: *mut f64, incX: c_int);
  pub fn cblas_dgemv(
      Order: CBLAS_ORDER,
      TransA: CBLAS_TRANSPOSE,
      M: c_int, N: c_int,
      alpha: f64,
      A: *const f64, lda: c_int,
      X: *const f64, incX: c_int,
      beta: f64,
      Y: *mut f64, incY: c_int);
  pub fn cblas_dgemm(
      Order: CBLAS_ORDER,
      TransA: CBLAS_TRANSPOSE,
      TransB: CBLAS_TRANSPOSE,
      M: c_int, N: c_int, K: c_int,
      alpha: f64,
      A: *const f64, lda: c_int,
      B: *const f64, ldb: c_int,
      beta: f64,
      C: *mut f64, ldc: c_int);
}
