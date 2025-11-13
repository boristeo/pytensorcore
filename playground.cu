#include <cuda_fp16.h>
#include <stdio.h>

#define WARP_SIZE 32
#define Mmma 16
#define Nmma 8
#define Kmma 16

constexpr int M = 16, N = 8, K = 16;

// **** BEGIN GENERATED ****

extern "C"
__global__ void matmul_fp16_fp32(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, float* __restrict__ D) {
    unsigned int tid = threadIdx.x;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bz = blockIdx.z;
    half a_frag[8];
    half b_frag[4];
    float d_frag[4] = {0.0f,0.0f,0.0f,0.0f};

    constexpr const int APAD = 0; // 8
    constexpr const int BPAD = 0; // 8
    __shared__ half s_a[M*(K + APAD)];
    __shared__ half s_b[K*(N + BPAD)];
    half* s_a_addr = (half*)__cvta_generic_to_shared(s_a);
    half* s_b_addr = (half*)__cvta_generic_to_shared(s_b);

    // Synchronous
    //s_a[tid*8+0] = A[tid*8+0];  // (16x16)/32thr -> each thread loads 8 elems (8x16b->4x32b)
    //s_a[tid*8+1] = A[tid*8+1];
    //s_a[tid*8+2] = A[tid*8+2];
    //s_a[tid*8+3] = A[tid*8+3];
    //s_a[tid*8+4] = A[tid*8+4];
    //s_a[tid*8+5] = A[tid*8+5];
    //s_a[tid*8+6] = A[tid*8+6];
    //s_a[tid*8+7] = A[tid*8+8];
    //s_b[tid*4+0] = B[tid*4+0];  // (16x8)/32thr -> each thread loads 4 elems (4x16b->2x32b)
    //s_b[tid*4+1] = B[tid*4+1];
    //s_b[tid*4+2] = B[tid*4+2];
    //s_b[tid*4+3] = B[tid*4+3];


    //i=[0,7]: s_a[tid*8+i] = A[tid*8+i];  // (16x16)/32thr -> each thread loads 8 elems (8x16b->1x128b)
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"
      :
      : "r"((int32_t)(&s_a_addr[tid*8]))
      , "l"(&A[tid*8])
    );
    //i=[0,3]: s_b[tid*4+i] = B[tid*4+i];  // (16x8)/32thr -> each thread loads 4 elems (4x16b->1x64b)
    asm volatile("cp.async.ca.shared.global [%0],[%1],8;\n"
      :
      : "r"((int32_t)(&s_b_addr[tid*4]))
      , "l"(&B[tid*4])
    );
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);

    __syncthreads();





    for (int i = 0; i < 1; ++i) {
      a_frag[0] = s_a[(by)*0+(0+(tid/4))*16+(i)*0+(0+(tid%4)*2)*1];
      a_frag[1] = s_a[(by)*0+(0+(tid/4))*16+(i)*0+(1+(tid%4)*2)*1];
      a_frag[2] = s_a[(by)*0+(8+(tid/4))*16+(i)*0+(0+(tid%4)*2)*1];
      a_frag[3] = s_a[(by)*0+(8+(tid/4))*16+(i)*0+(1+(tid%4)*2)*1];
      a_frag[4] = s_a[(by)*0+(0+(tid/4))*16+(i)*0+(8+(tid%4)*2)*1];
      a_frag[5] = s_a[(by)*0+(0+(tid/4))*16+(i)*0+(9+(tid%4)*2)*1];
      a_frag[6] = s_a[(by)*0+(8+(tid/4))*16+(i)*0+(8+(tid%4)*2)*1];
      a_frag[7] = s_a[(by)*0+(8+(tid/4))*16+(i)*0+(9+(tid%4)*2)*1];

      b_frag[0] = s_b[(i)*0+(0+(tid%4)*2)*8+(bx)*0+(tid/4)*1];
      b_frag[1] = s_b[(i)*0+(1+(tid%4)*2)*8+(bx)*0+(tid/4)*1];
      b_frag[2] = s_b[(i)*0+(8+(tid%4)*2)*8+(bx)*0+(tid/4)*1];
      b_frag[3] = s_b[(i)*0+(9+(tid%4)*2)*8+(bx)*0+(tid/4)*1];


      // --- MMA PTX ---
      unsigned int* a_int = reinterpret_cast<unsigned int *>(a_frag);
      unsigned int* b_int = reinterpret_cast<unsigned int *>(b_frag);
      unsigned int* d_int = reinterpret_cast<unsigned int *>(d_frag);
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(d_int[0]),"+r"(d_int[1]),"+r"(d_int[2]),"+r"(d_int[3])
        : "r"(a_int[0]),"r"(a_int[1]),"r"(a_int[2]),"r"(a_int[3])
        , "r"(b_int[0]),"r"(b_int[1])
      );

    }

    D[(by)*0+(0+(tid/4))*8+(bx)*0+(0+(tid%4)*2)*1] = d_frag[0];
    D[(by)*0+(0+(tid/4))*8+(bx)*0+(1+(tid%4)*2)*1] = d_frag[1];
    D[(by)*0+(8+(tid/4))*8+(bx)*0+(0+(tid%4)*2)*1] = d_frag[2];
    D[(by)*0+(8+(tid/4))*8+(bx)*0+(1+(tid%4)*2)*1] = d_frag[3];

}


// **** END GENERATED ****

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



    unsigned int TILES_M = int(ceil((float)M / Mmma));
    unsigned int TILES_N = int(ceil((float)N / Nmma));
    unsigned int TILES_K = int(ceil((float)K / Kmma));

    // Launch: 1 block, WARP_SIZE threads
    matmul_fp16_fp32<<<{TILES_M, TILES_N, 1}, WARP_SIZE>>>(d_A, d_B, nullptr, d_D);
    cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Output: each value should be 16.0
    #define DEBUG 1
    if (DEBUG) {
      for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++)
              printf("%6.1f ", h_D[i * N + j]);
          printf("\n");
      }
    }
    
    for (int i = 0; i < N; i++) if (h_D[i] != float(K)) {puts("WRONG"); break;}

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    free(h_A); free(h_B); free(h_D);
    return 0;
}

