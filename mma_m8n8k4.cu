#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
  std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
            << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1);} } while(0)

// ---------------------------------------------------------------------------
// Kernel:  one warp does 4 8x8x4 Tensor Core matmuls in FP16
// ---------------------------------------------------------------------------
__global__ void mma_m8n8k4_fp16_full(const half *A, const half *B, half *D)
{
    unsigned lane = threadIdx.x & 31;

    // Each lane has 4 32-bit registers = 8 half accumulators
    uint32_t d[4];

    A += 4 * (lane % 4) + (lane >= 16 ? 16 : 0);

    B += 8 * (lane % 4) + (lane >= 16 ? 4 : 0);

    asm volatile(
        "{\n"
        ".reg .b32 a<2>, b<2>, c<4>, d<4>;\n"
        ".reg .u64 ra, rb;\n"
        "mov.u64 ra, %4;\n"
        "mov.u64 rb, %5;\n"

        // load A (8x4 half -> 2x32-bit words)
        "ld.global.b32 a0, [ra + 0];\n"
        "ld.global.b32 a1, [ra + 4];\n"

        // load B (4x8 half -> 2x32-bit words)
        "ld.global.b32 b0, [rb + 0];\n"
        "ld.global.b32 b1, [rb + 4];\n"

        // zero accumulators
        "mov.b32 c0, 0; mov.b32 c1, 0; mov.b32 c2, 0; mov.b32 c3, 0;\n"

        // MMA Tensor Core op
        "mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
        "{%0,%1,%2,%3}, {a0,a1}, {b0,b1}, {c0,c1,c2,c3};\n"

        // move outputs to C vars so C++ can see them
        //"mov.b32 %0, d0;\n"
        //"mov.b32 %3, d1;\n"
        //"mov.b32 %4, d2;\n"
        //"mov.b32 %5, d3;\n"
        "}\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "l"(A), "l"(B)
    );

    // Now scatter each thread's 8 halves to their correct (row,col)
    int row = (lane < 16) ? (lane % 4) : (lane % 4) + 4;

    // reinterpret 32-bit regs as two half values
    half2 *d_half2 = reinterpret_cast<half2*>(d);
    half vals[8];
    for (int i = 0; i < 4; ++i) {
        half2 h2 = d_half2[i];
        vals[2*i+0] = __low2half(h2);
        vals[2*i+1] = __high2half(h2);
    }

    // Write 8 columns for this row
    for (int ci = 0; ci < 8; ++ci) {
        int col = ci;
        int idx = row * 8 + col;
        D[idx] = vals[ci];
    }
}

// ---------------------------------------------------------------------------
// Host-side harness
// ---------------------------------------------------------------------------
int main() {
    const int M=8, N=8, K=4;
    const int sizeA = M*K;
    const int sizeB = K*N;
    const int sizeD = M*N;

    std::vector<half> hA(sizeA), hB(sizeB), hD(sizeD);

    // Fill A,B with 1.0
    //for (int i=0; i<sizeA; ++i) hA[i] = __float2half(1.0f);
    //for (int i=0; i<sizeB; ++i) hB[i] = __float2half((float)i);

    //for (int i=0; i<sizeA; ++i) hA[i] = __float2half(0.0f);
    //for (int i=0; i<sizeB; ++i) hB[i] = __float2half(1.0f);
    //hB[0] = __float2half(1.0f);

    hA = {
      1, 0, 1, 0,
      0, 1, 0, 1,
      1, 0, 1, 0,
      0, 1, 0, 1,
      1, 0, 1, 0,
      0, 1, 0, 1,
      1, 0, 1, 0,
      0, 1, 0, 1,
      };
    hB = {
      1, 1, 0, 0, 0, 0, 1, 1,
      0, 0, 1, 1, 1, 1, 0, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
      1, 1, 0, 0, 0, 0, 1, 1,
      };

    half *dA,*dB,*dD;
    CUDA_CHECK(cudaMalloc(&dA,sizeA*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB,sizeB*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dD,sizeD*sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dA,hA.data(),sizeA*sizeof(half),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB.data(),sizeB*sizeof(half),cudaMemcpyHostToDevice));

    mma_m8n8k4_fp16_full<<<1,32>>>(dA,dB,dD);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hD.data(),dD,sizeD*sizeof(half),cudaMemcpyDeviceToHost));

    std::cout << "Result (8x8):\n";
    for (int r=0;r<M;++r){
        for (int c=0;c<N;++c){
            std::cout << __half2float(hD[r*N+c]) << " ";
        }
        std::cout << "\n";
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return 0;
}

