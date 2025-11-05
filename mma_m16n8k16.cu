#include <cuda_fp16.h>
#include <stdio.h>

// Constants for fragment sizing and MMA shape
#define M 16
#define N 8
#define K 16
#define WARP_SIZE 32

// CUDA kernel: MMA using explicit global loads and inline PTX
__global__ void mma_m16n8k16_global(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ D) {
    // Lane and group assignment within the warp
    unsigned int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    unsigned int groupID = lane_id >> 2;
    unsigned int threadID_in_group = lane_id & 3;

    // Output accumulator fragments (FP32)
    float c_frag[4] = {0.f, 0.f, 0.f, 0.f};
    float d_frag[4];

    // MMA expects 2 half per 32b register (half2), so there are 4 half2 for A, 2 for B
    half2 a_frag[4];
    half2 b_frag[2];

    // --- Fill A fragment (4 regs, 2 half each) ---
    for (int frag = 0; frag < 4; ++frag) {
        // For each half in this half2
        half h0, h1;
        for (int subidx = 0; subidx < 2; ++subidx) {
            int ai = frag * 2 + subidx;
            unsigned int row = ( (ai < 2) || (ai >= 4 && ai < 6) ) ? groupID : (groupID + 8);
            unsigned int col = (threadID_in_group * 2) + (ai & 1);
            if (ai >= 4) col += 8;
            int index = row * K + col;
            half value = __float2half(0.0f);
            if(row < M && col < K)
                value = A[index];
            if(subidx == 0) h0 = value;
            else h1 = value;
        }
        a_frag[frag] = __halves2half2(h0, h1);
    }

    // --- Fill B fragment (2 regs, 2 half each) ---
    for (int frag = 0; frag < 2; ++frag) {
        half h0, h1;
        for (int subidx = 0; subidx < 2; ++subidx) {
            int bi = frag * 2 + subidx;
            unsigned int row = (threadID_in_group * 2) + (bi & 1);
            if (bi >= 2) row += 8;
            unsigned int col = groupID;
            int index = row * N + col;
            half value = __float2half(0.0f);
            if(row < K && col < N)
                value = B[index];
            if(subidx == 0) h0 = value;
            else h1 = value;
        }
        b_frag[frag] = __halves2half2(h0, h1);
    }

    // Cast to unsigned for inline PTX
    unsigned int a_int[4], b_int[2];
    #pragma unroll
    for(int i = 0; i < 4; ++i)
        a_int[i] = reinterpret_cast<unsigned int &>(a_frag[i]);
    #pragma unroll
    for(int i = 0; i < 2; ++i)
        b_int[i] = reinterpret_cast<unsigned int &>(b_frag[i]);

    // --- MMA PTX ---
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
        : "r"(a_int[0]), "r"(a_int[1]), "r"(a_int[2]), "r"(a_int[3]),
          "r"(b_int[0]), "r"(b_int[1]),
          "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3])
    );

    // --- Write results by fragment-to-tile mapping ---
    for (int i = 0; i < 4; i++) {
        unsigned int row = (i < 2) ? groupID : (groupID + 8);
        unsigned int col = (threadID_in_group * 2) + (i & 1);
        if (row < M && col < N) {
            unsigned int idx = row * N + col;
            D[idx] = d_frag[i];
        }
    }
}

int main() {
    // Host allocation and initialization with all 1s (FP16)
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_D = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(1.0f);

    // Device mem
    half *d_A, *d_B;
    float *d_D;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_D, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_D, 0, M * N * sizeof(float));

    // Launch: 1 block, WARP_SIZE threads
    mma_m16n8k16_global<<<1, WARP_SIZE>>>(d_A, d_B, d_D);
    cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Output: each value should be 16.0
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%6.1f ", h_D[i * N + j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    free(h_A); free(h_B); free(h_D);
    return 0;
}

