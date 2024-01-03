// [1, m] * [k, m] => [1, k]
// A: [1, m]
// B: [k, m]
// C: [1, k]
// bias: [1, k]
// scales: [k, 1]
// mins: [k, 1]

// ALG1: 每一个线程计算一个 VV, 一共 K 个线程
__kernel void GemvFloatInt4NoZero(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[256];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];
  for (int i = tid; i < (m >> 1); i += tsz) {
    uchar now = B[gid * (m >> 1) + i];
    sdata[tid] += (A[i * 2] * (minv + (now >> 4))) + (A[i * 2 + 1] * (minv + (now & 15)));
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid < 64) {
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    if (tid < 1) sdata[tid] += sdata[tid + 1];
  }

  // barrier(CLK_LOCAL_MEM_FENCE);

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}

__kernel void GemvFloatInt4NoZero3(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[256];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];

  // Loop unrolling
  for (int i = tid; i < (m >> 2); i += tsz) {
    uchar now = B[gid * (m >> 2) + i];
    sdata[tid] += (A[i * 4] * (minv + (now >> 4))) + (A[i * 4 + 1] * (minv + (now & 15)));
    now = B[gid * (m >> 2) + i + 1];
    sdata[tid] += (A[i * 4 + 2] * (minv + (now >> 4))) + (A[i * 4 + 3] * (minv + (now & 15)));
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Optimized reduction operation
  for (int s = tsz / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}


__kernel void GemvFloatInt4NoZero4(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[256];
  __local float adata[256];
  __local uchar bdata[256];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];

  for (int i = tid; i < (m >> 1); i += tsz) {
    adata[tid] = A[i * 2];
    bdata[tid] = B[gid * (m >> 1) + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    uchar now = bdata[tid];
    sdata[tid] += (adata[tid] * (minv + (now >> 4))) + (adata[tid + 1] * (minv + (now & 15)));

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Optimized reduction operation
  for (int s = tsz / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}



__kernel void GemvFloatInt4NoZero2(__global float *A, __global uchar *B,
                                  __global float *C, __global float *bias,
                                  __global float *scales, __global float *mins,
                                  const int m, const int k) {
  int tid = get_local_id(1);
  int tsz = get_local_size(1);

  int gid = get_global_id(0);
  __local float sdata[LOCAL_SIZE];

  sdata[tid] = 0.0f;
  float minv = mins[gid] / scales[gid];
  for (int i = tid; i < (m >> 2); i += tsz) {
    uchar2 now = vload2(gid * (m >> 1) + (i << 1), B);
    float4 A_float = vload4(0, A + (i << 2));
    float4 B_float = minv + (float4)(now.x >> 4, now.x & 0xf, now.y >> 4, now.y & 0xf);
    sdata[tid] += dot(A_float, B_float);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (tid < 64) {
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    if (tid < 1) sdata[tid] += sdata[tid + 1];
  }

  // barrier(CLK_LOCAL_MEM_FENCE);

  if (tid == 0) {
    C[gid] = sdata[0] * scales[gid] + bias[gid];
  }
}


__kernel void gemv_quantized(__global uchar* A, __global float* x, __global float* y, int k, int m, __global float* scale, __global float* min) {
    int gid = get_global_id(0);
    if (gid < k) {
      float8 min_value = (float8)(min[gid], min[gid], min[gid], min[gid], min[gid], min[gid], min[gid], min[gid]);
      float8 scale_value = (float8)(scale[gid], scale[gid], scale[gid], scale[gid], scale[gid], scale[gid], scale[gid], scale[gid]);
      float8 minv = min_value / scale_value;
      float dot_product = 0.0f;
      for (int j = 0; j < m; j += 8) {
        uchar4 A_int = vload4(gid * (m >> 1) + (j >> 1), A);
        float8 A_float = minv + 
                          (float8)(A_int.s0 >> 4, A_int.s0 & 0xF, A_int.s1 >> 4, A_int.s1 & 0xF, 
                                   A_int.s2 >> 4, A_int.s2 & 0xF, A_int.s3 >> 4, A_int.s3 & 0xF);
        float8 x_float = vload8(j, x);
        dot_product += dot(A_float.lo, x_float.lo) + dot(A_float.hi, x_float.hi);
      }
      y[gid] = dot_product * scale[gid];
    }
}
// 调用代码
// cl::Kernel kernel{program, "gemv_quantized"};
// int idx = 0;
// ret |= kernel.setArg(idx++, bufferA);
// ret |= kernel.setArg(idx++, bufferX);
// ret |= kernel.setArg(idx++, bufferY);
// ret |= kernel.setArg(idx++, m);
// ret |= kernel.setArg(idx++, n);
// ret |= kernel.setArg(idx++, bufferScale);
// ret |= kernel.setArg(idx++, bufferMin);
// ret |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(m));


// __kernel void gemv_kernel(
//     __global uchar* matrix,
//     __global float* vector,
//     __global float* result,
//     __global float* bias,
//     int k,
//     int m,
//     __global float* scale,
//     __global float* min
// ) {
//     int gid = get_global_id(0);
//     int lid = get_local_id(0);

//     // Local memory to store a tile of the matrix
//     __local uchar localMatrixTile[LOCAL_SIZE][LOCAL_SIZE];

//     // Load the matrix tile into local memory
//     for (int i = lid; i < m; i += LOCAL_SIZE) {
//         localMatrixTile[lid][i] = matrix[gid * m + i];
//     }

//     barrier(CLK_LOCAL_MEM_FENCE);

//     // Compute the result using the matrix tile, vector, and bias
//     float sum = 0.0f;
//     for (int i = 0; i < m; ++i) {
//         uchar matrix_value = localMatrixTile[lid][i];
//         float matrix_float = scale[gid] * (float)matrix_value + min[gid];
//         sum += matrix_float * vector[i];
//     }

//     // Use work-group reduction to compute the final result
//     sum = work_group_reduce_add(sum);

//     // Write the result to global memory
//     if (lid == 0) {
//         result[gid] = sum + bias[gid];
//     }
// }
