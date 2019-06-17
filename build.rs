#[cfg(target_os = "linux")]
fn linux_main() {
  println!("cargo:rustc-link-lib=dylib=blas");
}

#[cfg(target_os = "macos")]
fn macos_main() {
  println!("cargo:rustc-link-lib=framework=Accelerate");
}

fn main() {
  #[cfg(target_os = "linux")]
  {
    linux_main();
  }
  #[cfg(target_os = "macos")]
  {
    macos_main();
  }
}
