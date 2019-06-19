extern crate sysblas;

// NB: These tests mostly exist to check that linking succeeded.

#[test]
fn test_dasum() {
  let x: Vec<f64> = vec![1.0, -2.0, 3.0];
  let r = unsafe { sysblas::cblas_dasum(
      x.len() as _,
      x.as_ptr(), 1) };
  assert_eq!(6.0, r);
}

#[test]
fn test_daxpy() {
  let x: Vec<f64> = vec![1.0, 2.0, 3.0];
  let mut y: Vec<f64> = vec![3.0, 2.0, 1.0];
  unsafe { sysblas::cblas_daxpy(
      x.len() as _,
      1.0,
      x.as_ptr(), 1,
      y.as_mut_ptr(), 1) };
  assert_eq!(&[4.0, 4.0, 4.0], y.as_slice());
}

#[test]
fn test_dcopy() {
  let x: Vec<f64> = vec![1.0, 2.0, 3.0];
  let mut y: Vec<f64> = vec![3.0, 2.0, 1.0];
  unsafe { sysblas::cblas_dcopy(
      x.len() as _,
      x.as_ptr(), 1,
      y.as_mut_ptr(), 1) };
  assert_eq!(&[1.0, 2.0, 3.0], y.as_slice());
}

#[test]
fn test_ddot() {
  let x: Vec<f64> = vec![1.0, 2.0, 3.0];
  let y: Vec<f64> = vec![3.0, 2.0, 1.0];
  let r = unsafe { sysblas::cblas_ddot(
      x.len() as _,
      x.as_ptr(), 1,
      y.as_ptr(), 1) };
  assert_eq!(10.0, r);
}

#[test]
fn test_dnrm2() {
  let x: Vec<f64> = vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0];
  let r = unsafe { sysblas::cblas_dnrm2(
      x.len() as _,
      x.as_ptr(), 1) };
  assert_eq!(10.0, r);
}

#[test]
fn test_dscal() {
  let mut x: Vec<f64> = vec![1.0, 2.0, 3.0];
  unsafe { sysblas::cblas_dscal(
      x.len() as _,
      2.0,
      x.as_mut_ptr(), 1) };
  assert_eq!(&[2.0, 4.0, 6.0], x.as_slice());
}

#[test]
fn test_dgemv() {
  let a: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
  let x: Vec<f64> = vec![1.0, 2.0, 3.0];
  let mut y: Vec<f64> = vec![0.0, 0.0, 0.0];
  unsafe { sysblas::cblas_dgemv(
      sysblas::CblasColMajor,
      sysblas::CblasNoTrans,
      3, 3,
      1.0,
      a.as_ptr(), 3,
      x.as_ptr(), 1,
      0.0,
      y.as_mut_ptr(), 1) };
  assert_eq!(&[2.0, 3.0, 1.0], y.as_slice());
}

#[test]
fn test_dgemm() {
  let a: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
  let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
  let mut c: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  unsafe { sysblas::cblas_dgemm(
      sysblas::CblasColMajor,
      sysblas::CblasNoTrans,
      sysblas::CblasNoTrans,
      3, 3, 3,
      1.0,
      a.as_ptr(), 3,
      b.as_ptr(), 3,
      0.0,
      c.as_mut_ptr(), 3) };
  assert_eq!(&[2.0, 3.0, 1.0, 5.0, 6.0, 4.0, 8.0, 9.0, 7.0], c.as_slice());
}
