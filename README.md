# pytensorcore

Recently decided that I want to get familiar with GPUs. Turns out my vague understanding of CUDA programming is not actually too relevant to the new, actually performant ways to do big matmuls.

Throughout the past decade of GPU development, it seems to me they've mainly been bolting specialized little hardware blocks to various locations on chip, and getting the advertised FLOPS relies on using these new blocks optimally.

My goal is to start with the tensor cores. Try to get a nice, scalable way to run any size matrix multiply at close to the theoretical FLOPS. Here I'll assume nice memory layout, or maybe forget about global loads altogether.

Later, I will deal with memory access. From what I've read, I suspect people who actually write kernels for a living would start with this because for them, when the data's in the right place compute is trivial. For now I don't know what the access patterns are going to be yet, and I want to set up a good workflow to profile sufficiently intense compute so that bad scheduling visibly makes it run worse.

