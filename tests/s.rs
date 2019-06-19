extern crate sysblas;

// NB: These tests mostly exist to check that linking succeeded.

#[test]
fn test_sasum() {
  let x: Vec<f32> = vec![1.0, -2.0, 3.0];
  let r = unsafe { sysblas::cblas_sasum(
      x.len() as _,
      x.as_ptr(), 1) };
  assert_eq!(6.0, r);
}

#[test]
fn test_saxpy() {
  let x: Vec<f32> = vec![1.0, 2.0, 3.0];
  let mut y: Vec<f32> = vec![3.0, 2.0, 1.0];
  unsafe { sysblas::cblas_saxpy(
      x.len() as _,
      1.0,
      x.as_ptr(), 1,
      y.as_mut_ptr(), 1) };
  assert_eq!(&[4.0, 4.0, 4.0], y.as_slice());
}

#[test]
fn test_scopy() {
  let x: Vec<f32> = vec![1.0, 2.0, 3.0];
  let mut y: Vec<f32> = vec![3.0, 2.0, 1.0];
  unsafe { sysblas::cblas_scopy(
      x.len() as _,
      x.as_ptr(), 1,
      y.as_mut_ptr(), 1) };
  assert_eq!(&[1.0, 2.0, 3.0], y.as_slice());
}

#[test]
fn test_sdot() {
  let x: Vec<f32> = vec![1.0, 2.0, 3.0];
  let y: Vec<f32> = vec![3.0, 2.0, 1.0];
  let r = unsafe { sysblas::cblas_sdot(
      x.len() as _,
      x.as_ptr(), 1,
      y.as_ptr(), 1) };
  assert_eq!(10.0, r);
}

#[test]
fn test_snrm2() {
  let x: Vec<f32> = vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0];
  let r = unsafe { sysblas::cblas_snrm2(
      x.len() as _,
      x.as_ptr(), 1) };
  assert_eq!(10.0, r);
}

#[test]
fn test_sscal() {
  let mut x: Vec<f32> = vec![1.0, 2.0, 3.0];
  unsafe { sysblas::cblas_sscal(
      x.len() as _,
      2.0,
      x.as_mut_ptr(), 1) };
  assert_eq!(&[2.0, 4.0, 6.0], x.as_slice());
}

#[test]
fn test_sgemv() {
  let a: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
  let x: Vec<f32> = vec![1.0, 2.0, 3.0];
  let mut y: Vec<f32> = vec![0.0, 0.0, 0.0];
  unsafe { sysblas::cblas_sgemv(
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
fn test_sgemm() {
  let a: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
  let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
  let mut c: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  unsafe { sysblas::cblas_sgemm(
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
