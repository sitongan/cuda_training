#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

double random_double(void)
{
return 1.0;
//  return static_cast<double>(rand()) / RAND_MAX;
}


// Part 1 of 6: implement the kernel
__global__ void block_sum(const double *input,
                          double *per_block_results,
                          const size_t n)
{
  //fill me
  __shared__ double sdata[blockDim.x];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n){
    sdata[threadIdx.x] = input[i];
    __syncthreads();
    //atomicAdd(&per_block_results[blockIdx.x], sdata[threadIdx.x]);
    int totalThreads = blockDim.x;
    while(totalThreads >1){
      totalThreads = (totalThreads >> 1);
      if (threadIdx.x < totalThreads){
        sdata[threadIdx.x] += sdata[threadIdx.x + totalThreads];        
      }
    __syncthreads();
    }
  }
  __syncthreads();
    
    
  per_block_results[blockIdx.x] = sdata[0];
 
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  
  const int blockDim = 512;
  
  // create array of 256ki elements
  //const int num_elements = 1<<18;
  const int num_elements = 512;
  srand(time(NULL));
  // generate random input on the host
  std::vector<double> h_input(num_elements);
  for(int i = 0; i < h_input.size(); ++i)
  {
    h_input[i] = random_double();
  }

  const double host_result = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
  std::cerr << "Host sum: " << host_result << std::endl;

  //Part 1 of 6: move input to device memory
  double *d_input = 0;
  cudaMalloc((void**)&d_input, num_elements * sizeof(double) );
  cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(double), cudaMemcpyHostToDevice);
  
  // Part 1 of 6: allocate the partial sums: How much space does it need?
  double *d_partial_sums_and_total = 0;
  cudaMalloc((void**)&d_partial_sums_and_total, num_elements / blockDim * sizeof(double) );
  
  // Part 1 of 6: copy the result back to the host
  double *d_result = 0;
  double device_result = 0;
  cudaMalloc((void**)&d_result, 1 * sizeof(double));
  
  // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How much shared memory does it need?
  block_sum<<<num_elements / blockDim, blockDim>>>(d_input, d_partial_sums_and_total, num_elements);
  block_sum<<<1, blockDim>>>(d_partial_sums_and_total, d_result, num_elements / blockDim);
  
  // Part 1 of 6: compute the sum of the partial sums
  cudaMemcpy(&device_result, d_result, 1 * sizeof(double), cudaMemcpyDeviceToHost);


  std::cout << "Device sum: " << device_result << std::endl;

  // Part 1 of 6: deallocate device memory

  cudaFree(d_input);
  cudaFree(d_partial_sums_and_total);
  cudaFree(d_result);

  return 0;
}
