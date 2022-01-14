
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
 
#include "config.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}
                
// Add GPU kernel and functions
// HERE!!!

__global__ void calcHist(unsigned char *input, 
                         unsigned int *histogram,
                         const unsigned int height,
                         const unsigned int width){
    __shared__ unsigned int shared_hist[256];

    const int idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	const int location = y*(gridDim.x*TILE_SIZE)+x;


    if (idx_in_block < 256) 
        shared_hist[idx_in_block] = 0;

    __syncthreads();

    if (y < height && x < width) {
        atomicAdd(&histogram[input[location]], 1u);
    }

    __syncthreads();

    if (idx_in_block < 256) {
        unsigned int value = shared_hist[idx_in_block];
        if(value != 0)
            atomicAdd(&histogram[idx_in_block], value);
    }
}

__global__ void calcHistFreq(unsigned int *histogram) {
    __shared__ unsigned int hist[256];

    //this method will only be called with single 256 sized block
    const int tid = threadIdx.x;

    hist[tid] = (tid > 0 ? histogram[tid] : 0);

    //prefix sum
    for(int i = 1; i < 256; i <<= 1) {
        __syncthreads();
        if(tid >= i) {
            hist[tid] += hist[tid-i];
        }
    }
    __syncthreads();
    
    //multiply by grey scale
    histogram[tid] = ((float)hist[tid]/hist[255]) * 255.0;
}

__global__ void mapFromHist(unsigned char *input, 
                            unsigned int *histogram){

    const int location = (blockIdx.y*TILE_SIZE+threadIdx.y)*TILE_SIZE*gridDim.x+(blockIdx.x*TILE_SIZE+threadIdx.x);
    
    input[location] = histogram[input[location]];
}

__global__ void warmup(unsigned char *input, 
                       unsigned char *output){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
    output[location] = 0;

}

// NOTE: The data passed on is already padded
void gpu_function(unsigned char *data,  
                  unsigned int height, 
                  unsigned int width){
    
    unsigned char *input_gpu; //combine pointer to image and histogram

	const int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	const int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	const int XSize = gridXSize*TILE_SIZE;
	const int YSize = gridYSize*TILE_SIZE;
	
	const int size = XSize*YSize;

    const int histogram_size = 256;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu, 
        size*sizeof(unsigned char) + histogram_size*sizeof(unsigned int)));
	
    // Zero the histogram
    checkCuda(cudaMemset((unsigned int *)(input_gpu + size) , 
        0 , histogram_size*sizeof(unsigned int)));
	
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(unsigned char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	// Kernel Call
	#ifdef CUDA_TIMING
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
    //calculate the histogram bins
    calcHist<<<dimGrid, dimBlock>>>(input_gpu, 
                                    (unsigned int *)(input_gpu + size),
                                    height,
                                    width);

    //prefix sum, probability, and equalization
    calcHistFreq<<<1,256>>>((unsigned int *)(input_gpu + size));

    //map the new histogram back onto the old image
    mapFromHist<<<dimGrid, dimBlock>>>(input_gpu,
                                       (unsigned int *)(input_gpu + size));

    
    // From here on, no need to change anything
    checkCuda(cudaPeekAtLastError());                                     
    checkCuda(cudaDeviceSynchronize());
	
	#ifdef CUDA_TIMING
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			input_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));

    // Free resources and end the program
	checkCuda(cudaFree(input_gpu));
}

void gpu_warmup(unsigned char *data, 
                unsigned int height, 
                unsigned int width){
    
    unsigned char *input_gpu;
    unsigned char *output_gpu;
     
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
    const int histogram_size = 256;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu, 
        size*sizeof(unsigned char) + histogram_size*sizeof(unsigned int)));
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	
    // Zero the histogram
    checkCuda(cudaMemset((unsigned int *)(input_gpu + size) , 
        0 , histogram_size*sizeof(unsigned int)));
	
	
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
            
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
    // Execute algorithm
        
	dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    warmup<<<dimGrid, dimBlock>>>(input_gpu, 
                                  output_gpu);
    //calculate the histogram bins
    calcHist<<<dimGrid, dimBlock>>>(input_gpu, 
                                    (unsigned int *)(input_gpu + size),
                                    height,
                                    width);

    //prefix sum, probability, and equalization
    calcHistFreq<<<1,256>>>((unsigned int *)(input_gpu + size));

    //map the new histogram back onto the old image
    mapFromHist<<<dimGrid, dimBlock>>>(input_gpu,
                                       (unsigned int *)(input_gpu + size));

                                         
    checkCuda(cudaDeviceSynchronize());
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));

}

