# sysblas

The `sysblas` crate provides minimal CBLAS FFI bindings to the system BLAS.
On Linux, this is assumed to be `libblas.so`; for Debian-based systems, the
implementation of `libblas.so` can be selected via `update-alternatives`.
On OS X, this is just `Accelerate.framework`.
