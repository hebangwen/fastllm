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