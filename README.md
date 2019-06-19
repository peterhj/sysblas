# sysblas

The `sysblas` crate provides minimal CBLAS FFI bindings to the system BLAS.
On Linux, this is assumed to be `libblas.so`; for Debian-based systems, the
implementation of `libblas.so` can be selected via `update-alternatives`.
On OS X, this is just `Accelerate.framework`.

There are a bunch of other BLAS bindings in Rust;
[search crates.io](https://crates.io/search?q=blas) to get an idea.
Some of them are intended as part of a multi-crate workflow separating
bindings and sources
(c.f. <https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki>).
Others use Cargo features to specify a BLAS implementation of choice
(c.f. <https://github.com/blas-lapack-rs/blas-src>).
`sysblas` is less powerful but more no-frills: it assumes you just have
some sort of system BLAS installation, you want to link to it, and you
want a minimal set of FFI bindings to it.
