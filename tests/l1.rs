extern crate sysblas;

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
