#include <cuda_runtime.h>
#include <stdio.h>
#include<iostream>
#include<math.h>
#include "../../../../../usr/include/crt/common_functions.h"
#include <string.h>

#define ull unsigned long long int

__global__ void multiply( unsigned long long int * op_mn,  int rows,  int cols,  int *kernel,  int kernel_row,  int kernel_col,  int *mn,  int m_col){

    int total_cells = rows * cols;
    int start_cell = (blockIdx.x*blockDim.x + threadIdx.x);
        if(total_cells <= start_cell) 
			return;
        int output_i = start_cell / cols;
	int output_j = start_cell % cols;
      
           for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
           {
                int kernel_j = 0;
               for (; kernel_j <= kernel_col - 4; kernel_j += 4)
               {
                   int input_i = output_i + kernel_i;
                   int input_j = output_j + kernel_j;
                   int op_index = output_i * cols + output_j;
                   int in_index = input_i * m_col + input_j;
                   int kernel_index = kernel_i * kernel_col + kernel_j;


                   op_mn[op_index] += mn[in_index] * kernel[kernel_index] + mn[in_index + 1] * kernel[kernel_index + 1] + mn[in_index + 2] * kernel[kernel_index + 2] + mn[in_index + 3] * kernel[kernel_index + 3];
               }


               // Handle the remainder of the loop
               for (int kernel_j = kernel_col - (kernel_col % 4); kernel_j < kernel_col; kernel_j++)
               {
                   int input_i = output_i + kernel_i;
                   int input_j = output_j + kernel_j;
                   op_mn[output_i * cols + output_j] += mn[input_i * m_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
               }
           }
}

// Fill in this function
void gpuThread(int input_row, int input_col, int *input, 
                int kernel_row, int kernel_col, int *kernel,
                int output_row, int output_col, ull *output) 
{

	 memset(output, 0, output_row*output_col);


   // creating padded input
   int padded_row = input_row + kernel_row;
   int padded_col = input_col + kernel_col;
   int *padded_input = new int[padded_row * padded_col];


   // filling padded input with input array and then padding to remove mod
   //[1][]
   //[][]
   for (int i = 0; i < input_row; i++)
   {
       int padded_i = i * padded_col;
       int input_i = i * input_col;
       for (int j = 0; j < input_col; j++)
       {
           padded_input[padded_i + j] = input[input_i + j];
       }
   }


   //[1][]
   //[1][]
   for (int i = 0; i < kernel_row; i++)
   {
       int padded_i = (i + input_row) * padded_col;
       int input_i = i * input_col;
       for (int j = 0; j < input_col; j++)
       {
           padded_input[padded_i + j] = input[input_i + j];
       }
   }


//    [1][1]
//    [1][1]
   for (int i = 0; i < padded_row; i++)
   {
       int padded_i = i * padded_col;
       int padded_i2 = padded_i + input_col;
       for (int j = 0; j < kernel_col; j++)
       {
           padded_input[padded_i2 + j] = padded_input[padded_i + j];
       }
   }


   // now dividing the padded_input into 4 matrices
   int even_row = (padded_row + 1) / 2;
   int even_col = (padded_col + 1) / 2;
   int odd_row = (padded_row) / 2;
   int odd_col = (padded_col) / 2;


   // m1- even_even
   int *m1 = new int[even_row * even_col];
   // m2- odd_even
   int *m2 = new int[odd_row * even_col];
   // m3- even_odd
   int *m3 = new int[even_row * odd_col];
   // m4- ODD_odd
   int *m4 = new int[odd_row * odd_col];


   for (int i = 0; i < even_row; i++)
   {
       int m1_i = i * even_col;
       int padded_i = i * 2 * padded_col;
       for (int j = 0; j < even_col; j++)
       {
           m1[m1_i + j] = padded_input[padded_i + j * 2];
       }
       int m3_i = i * odd_col;
       for (int j = 0; j < odd_col; j++)
       {
           m3[m3_i + j] = padded_input[padded_i + j * 2 + 1];
       }
   }


   for (int i = 0; i < odd_row; i++)
   {
       int m2_i = i * even_col;
       int padded_i = (1 + i * 2) * padded_col;
       for (int j = 0; j < even_col; j++)
       {
           m2[m2_i + j] = padded_input[padded_i + j * 2];
       }
       int m4_i = i * odd_col;
       for (int j = 0; j < odd_col; j++)
       {
           m4[m4_i + j] = padded_input[padded_i + j * 2 + 1];
       }
   }


   // now dividing the output into 4 matrices
   int op_even_row = (output_row + 1) / 2;
   int op_even_col = (output_col + 1) / 2;
   int op_odd_row = (output_row) / 2;
   int op_odd_col = (output_col) / 2;


   // m1- even_even
   unsigned long long int *op_m1 = new unsigned long long int[op_even_row * op_even_col];
   // m2- odd_even
   unsigned long long int *op_m2 = new unsigned long long int[op_odd_row * op_even_col];
   // m3- even_odd
   unsigned long long int *op_m3 = new unsigned long long int[op_even_row * op_odd_col];
   // m4- ODD_odd
   unsigned long long int *op_m4 = new unsigned long long int[op_odd_row * op_odd_col];


    cudaFree(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  

	int *d_input1, *d_kernel;
	int *d_input2;
	int *d_input3;
	int *d_input4;
	ull *d_output1;
	ull *d_output2;
	ull *d_output3;
	ull *d_output4;
	cudaMalloc((void**)&d_input1, sizeof(int) * even_row * even_col);
	cudaMalloc((void**)&d_input2, sizeof(int) * odd_row * even_col);
	cudaMalloc((void**)&d_input3, sizeof(int) * even_row * odd_col);
	cudaMalloc((void**)&d_input4, sizeof(int) * odd_row * odd_col);
    cudaMalloc((void**)&d_kernel, sizeof(int) * kernel_row * kernel_col);
	cudaMalloc((void**)&d_output1, sizeof(ull) * op_even_row * op_even_col);
	cudaMalloc((void**)&d_output2, sizeof(ull) * op_odd_row * op_even_col);
	cudaMalloc((void**)&d_output3, sizeof(ull) * op_even_row * op_odd_col);
	cudaMalloc((void**)&d_output4, sizeof(ull) * op_odd_row * op_odd_col);
	cudaMemcpy(d_input1, m1, sizeof(int) * even_row * even_col, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input2, m2, sizeof(int) * odd_row * even_col, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input3, m3, sizeof(int) * even_row * odd_col, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input4, m4, sizeof(int) * odd_row * odd_col, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernel, sizeof(int) * kernel_row*kernel_col, cudaMemcpyHostToDevice);
	
        //m1
        int maxThreadsNeeded1 = op_even_row * op_even_col;
        int blocksWeCreate1 = ceil((double)maxThreadsNeeded1 / deviceProp.maxThreadsPerBlock);
        dim3 threadsPerBlock1(deviceProp.maxThreadsPerBlock);
		dim3 numThreadBlocks1(blocksWeCreate1); 
        multiply<<<numThreadBlocks1, threadsPerBlock1>>>(d_output1, op_even_row, op_even_col, d_kernel, kernel_row, kernel_col, d_input1, even_col);


		//m2
		int maxThreadsNeeded2 = op_odd_row * op_even_col;
        int blocksWeCreate2 = ceil((double)maxThreadsNeeded2 / deviceProp.maxThreadsPerBlock);
        dim3 threadsPerBlock2(deviceProp.maxThreadsPerBlock);
		dim3 numThreadBlocks2(blocksWeCreate2); 
	   	multiply<<<numThreadBlocks2, threadsPerBlock2>>>(d_output2, op_odd_row, op_even_col, d_kernel, kernel_row, kernel_col, d_input2, even_col);


		//m3
		int maxThreadsNeeded3 = op_even_row * op_odd_col;
        int blocksWeCreate3 = ceil((double)maxThreadsNeeded3 / deviceProp.maxThreadsPerBlock);
        dim3 threadsPerBlock3(deviceProp.maxThreadsPerBlock);
		dim3 numThreadBlocks3(blocksWeCreate3); 
	   	multiply<<<numThreadBlocks3, threadsPerBlock3>>>(d_output3, op_even_row, op_odd_col, d_kernel, kernel_row, kernel_col, d_input3, odd_col);


		//m4
        
        int maxThreadsNeeded4 = op_odd_row * op_odd_col;
        int blocksWeCreate4 = ceil((double)maxThreadsNeeded4 / deviceProp.maxThreadsPerBlock);
        dim3 threadsPerBlock4(deviceProp.maxThreadsPerBlock);
		dim3 numThreadBlocks4(blocksWeCreate4); 
   		multiply<<<numThreadBlocks4, threadsPerBlock4>>>(d_output4, op_odd_row, op_odd_col, d_kernel, kernel_row, kernel_col, d_input4, odd_col);
 
	cudaMemcpy(op_m1, d_output1, sizeof(ull) * op_even_row*op_even_col, cudaMemcpyDeviceToHost);
	cudaMemcpy(op_m2, d_output2, sizeof(ull) * op_odd_row * op_even_col, cudaMemcpyDeviceToHost);
	cudaMemcpy(op_m3, d_output3, sizeof(ull) * op_even_row * op_odd_col, cudaMemcpyDeviceToHost);
	cudaMemcpy(op_m4, d_output4, sizeof(ull) * op_odd_row * op_odd_col, cudaMemcpyDeviceToHost);
	
	int x = 2 * output_col;


   // m1 row and m3 rows are same, two diff loops inside for m1 and m3 cols
   for (int i = 0; i < op_even_row; i++)
   {
       int op_m1_i = i * op_even_col;
       int op_m3_i = i * op_odd_col;
       int oi = i * x;


       // m1
       for (int j = 0; j < op_even_col; j++)
       {
           output[oi + j * 2] = op_m1[op_m1_i + j];
       }
       // m3
       for (int j = 0; j < op_odd_col; j++)
       {
           output[oi + j * 2 + 1] = op_m3[op_m3_i + j];
       }
   }


   for (int i = 0; i < op_odd_row; i++)
   {
       int op_m2_i = i * op_even_col;
       int op_m4_i = i * op_odd_col;
       int oi1 = (i * 2 + 1) * output_col;


       // m2
       for (int j = 0; j < op_even_col; j++)
       {
           output[oi1 + j * 2] = op_m2[op_m2_i + j];
       }
       // m4
       for (int j = 0; j < op_odd_col; j++)
       {
           output[oi1 + j * 2 + 1] = op_m4[op_m4_i + j];
       }
   }


}
