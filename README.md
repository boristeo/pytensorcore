# pytensorcore

This is just an educational exercise for myself to understand what primitives make writing high performance CUDA kernels easier.

Some examples of this have been trending lately, and I realized I need to learn more about architectural details of GPUs to tell who is being smart about it.

The goal is not to rely on builtin CUDA C++ functions that abstract away implementation details (like the actual tensor core dimensions, fragment loading, async data movements)

Currently my approach is generating C++, since that is still best for variable declarations, indexing calculations, and control flow. Where applicable, I insert inline PTX.

Ideally I'd like to approach some warp-level IR, that offers quick ways to change the distribution of work (compute and loading) across threads, while treating memory regions logically as arbitrary rank tensors, giving me flexibility to group elements regardless of physical layout.

The same principles will apply for full models. I've seen how to optimize neural nets from the top level down. At this lower one it feels exactly the same.

## Next steps:

- clean up syntax - make index generation also usable for raw pointers

- better profiling. Nsight is not working great for me right now

- implement more tricks from good handwritten kernels, find clean way to express them
