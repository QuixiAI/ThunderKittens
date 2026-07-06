# Backend Notes

QuixiCore CUDA is the NVIDIA implementation of the QuixiCore contract. CUDA
source may use CUDA C++, PTX, tensor cores, cooperative groups, and
architecture-specific variants when those choices remain behind the shared
operation semantics.

Current migration concern: many kernels live in legacy buckets such as
`elementwise`, `serving`, `quant`, and `tm_cuda`. New work should use the
semantic family layout documented in `docs/repository-structure.md`.
