
#include <cuda_runtime.h>
#include <stdio.h>
#include<iostream>
#include<math.h>

#define ull unsigned long long int

__global__ void Convolute(int input_row_size, int input_col_size, int *input,
		int kernel_row_size, int kernel_col_size, int *kernel,
		int output_row_size, int output_col_size, ull *output
		)
{
	int total_cells = output_row_size * output_col_size;
        int start_cell = (blockIdx.x*blockDim.x + threadIdx.x);
        if(start_cell >= total_cells) return;
        int output_row = start_cell / output_col_size;
	int output_col = start_cell % output_col_size;
        //int end_cell = (threadIdx.x == blockDim.x - 1) ? (blockIdx.x+1)*blockDim.x -1 : start_cell;
        //int output_row_start = start_cell / output_col_size;
	//int output_col_start = start_cell % output_col_size;
        //int output_row_end = (end_cell / output_col_size) % output_col_size;
	//int output_col_end = end_cell % output_col_size;
	//if(threadIdx.x == 0)printf("Block idx %d\tThread %d\nblock dim%d\n", blockIdx.x, threadIdx.x, blockDim.x);
	/*for(int output_row=output_row_start; output_row <= output_row_end ; ++output_row) {
	    //int row_half_addr = row * input_col_size;
	        for(int output_col=output_col_start; output_col <= output_col_end; ++output_col) {
	*/
        for(int kernel_row = 0; kernel_row< kernel_row_size; kernel_row++)
	{
		for(int kernel_col = 0; kernel_col< kernel_col_size; kernel_col++)
		{
			int input_row = (output_row + 2*kernel_row) % input_row_size;
			int input_col = (output_col + 2*kernel_col) % input_col_size;
			output[output_row * output_col_size + output_col] += input[input_row * input_col_size +input_col] 
				* kernel[kernel_row * kernel_col_size +kernel_col];
			/*if(output_row==0 && output_col==0) {
				printf("input[%d][%d]: %d\t", input_row, input_col, input[input_row * input_col_size + input_col]);
				printf("output[%d][%d]: %ul\t", output_row, output_col, output[output_row * output_col_size + output_col]);
				printf("kernel[%d][%d]: %d\t", kernel_row, kernel_col, kernel[kernel_row * kernel_col_size + kernel_col]);
				printf("\n");
                        }*/
		}
	}
        //        }
        //}
}

// Fill in this function
void gpuThread(int input_row_size, int input_col_size, int *input, 
                int kernel_row_size, int kernel_col_size, int *kernel,
                int output_row_size, int output_col_size, ull *output) 
{
    cudaFree(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  // Assumes you have a single GPU device
//     std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
//     std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
//     std::cout << "Max Blocks per SM: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
//     std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
//     std::cout << "Max Threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;

	//copy data to device
	int *dev_input, *dev_kernel;
	ull *dev_output;
	cudaMalloc((void**)&dev_input, sizeof(int) * input_row_size * input_col_size);
	cudaMalloc((void**)&dev_kernel, sizeof(int) * kernel_row_size * kernel_col_size);
	cudaMalloc((void**)&dev_output, sizeof(ull) * output_row_size * output_col_size);
	cudaMemcpy(dev_input, input, sizeof(int) * input_row_size*input_col_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_output, output, output_row*output_col, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kernel, kernel, sizeof(int) * kernel_row_size*kernel_col_size, cudaMemcpyHostToDevice);
	// Kernel invocation
	//int max_gpu = fjfj

        //mine
        int maxBlocks = deviceProp.multiProcessorCount * deviceProp.maxBlocksPerMultiProcessor;
        int maxBlocksX = maxBlocks * output_col_size / (output_row_size + output_col_size);
        int maxBlocksY = maxBlocks * output_row_size / (output_row_size + output_col_size);

        int maxBlocksXWeNeed = std::max((output_col_size) / deviceProp.warpSize, 1); // let's have at least warpSize number of threads in each block
        int numBlocksXWeCreate = std::min(maxBlocksX, maxBlocksXWeNeed);
        int threadNumPerBlockX = output_col_size / numBlocksXWeCreate;

        int maxBlocksYWeNeed = std::max((output_col_size) / deviceProp.warpSize, 1); // let's have at least warpSize number of threads in each block
        int numBlocksYWeCreate = std::min(maxBlocksY, maxBlocksYWeNeed);
        int threadNumPerBlockY = output_row_size / numBlocksYWeCreate;
        // std::cout << "numBlocksYCreated: " << numBlocksYWeCreate << "\tthreadNumPerBlockY: " << threadNumPerBlockY << "\n";
        // std::cout << "numBlocksXCreated: " << numBlocksXWeCreate << "\tthreadNumPerBlockX: " << threadNumPerBlockX << "\n";
        
        /*//abhay
        int no_ins_blocks = ceil((output_row * output_col)/(double)deviceProp.warpSize);
        int no_ins_blocks_perSM = ceil(no_ins_blocks/(double)deviceProp.multiProcessorCount);
        int no_ins_blocks_perThread = ceil(no_ins_blocks_perSM/(double)deviceProp.maxThreadsPerMultiProcessor);
        int no_thread_blocks = ceil(deviceProp.maxThreadsPerMultiProcessor/(double)deviceProp.maxThreadsPerBlock);
        int threads_per_threadBlock = ceil(deviceProp.maxThreadsPerMultiProcessor/(double)no_thread_blocks);
        std::cout << "no_thread_blocks: " << no_thread_blocks << "\tthreads_per_threadBlock: " << threads_per_threadBlock << "\n";
        */
        int maxThreadsNeeded = output_row_size * output_col_size;
        int blocksWeCreate = ceil((double)maxThreadsNeeded / deviceProp.maxThreadsPerBlock);
        dim3 threadsPerBlock(deviceProp.maxThreadsPerBlock);
	dim3 numBlocks(blocksWeCreate); //N / threadsPerBlock.x, N / threadsPerBlock.y)
        //std::cout << "starting kernel\n";                                   
	Convolute<<<numBlocks, threadsPerBlock>>>(input_row_size, input_col_size, dev_input, kernel_row_size, kernel_col_size, dev_kernel, output_row_size, output_col_size, dev_output);
	cudaDeviceSynchronize();
	cudaMemcpy(output, dev_output, sizeof(ull) * output_row_size*output_col_size, cudaMemcpyDeviceToHost);
}
