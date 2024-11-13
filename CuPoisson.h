#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "CudaReduction/CuReduction.h"

namespace cg = cooperative_groups;

template <class T, unsigned int blockSize>
__global__ void reduce5(T* g_idata, T* g_odata, unsigned int n) {
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

	T mySum = (i < n) ? g_idata[i]  : 0;
	if (i + blockSize < n) mySum += g_idata[i + blockSize];

	sdata[tid] = mySum;
	cg::sync(cta);

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256)) {
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128)) {
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64)) {
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32) {
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
			mySum += tile32.shfl_down(mySum, offset);
		}
	}

	// write result for this block to global mem
	if (cta.thread_rank() == 0)
	{
		g_odata[blockIdx.x] = mySum;
	}
}

struct CuPoisson
{
	unsigned int N = 0, k = 0;
	double eps = 0, res = 0, res0 = 0;
	double eps_iter = 1e-5;
	CudaReduction *CR = nullptr;
	dim3 gridDim, blockDim;
	size_t smem = 0, Nbytes = 0;
	cudaStream_t stream = 0;
	void* kernel = nullptr;
	void** args = nullptr;
	double* f_dev = nullptr, * f0_dev = nullptr;

	/**
	 * grid (n of blocks)
	 * block (n of threads per block)
	 * N, size of arrays
	 * kernel
	 * arguments, first two fields are necessary: {&ptr1, &ptr2, ... }
	 */
	CuPoisson(dim3 grid_, dim3 block_, unsigned int N_, void* kernel_, std::vector<void*> args_)
	{
		if (args_.size() < 2)
		{
			std::cout << "not enough args..." << std::endl;
			return;
		}

		gridDim = grid_;
		blockDim = block_;
		N = N_;
		Nbytes = sizeof(double) * N;
		kernel = kernel_;

		f_dev = (double*)args_[0];
		f0_dev = (double*)args_[1];

		set_args(args_);

		CR = new CudaReduction(f_dev, N, 512);
	}

	void solve()
	{

		k = 0;
		eps = 1.0;
		res = 0.0;
		res0 = 0.0;

		for (k = 0; k < 1000000; k++)
		{
			cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);

			res = CR->reduce(true);
			eps = abs(res - res0) / res0;
			res0 = res;

			swap_ptr(&f_dev, &f0_dev);

			if (eps < eps_iter)
			{
				break;
			}

			if (k % 1000 == 0) std::cout << "device k = " << k << ", eps = " << eps << std::endl;

		}
		std::cout << "device k = " << k << ", eps = " << eps << std::endl;
	}

	void run_kernel()
	{
		cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
	}

	private:
	void set_args(std::vector<void*> args_)
	{
		size_t n = args_.size();
		args = new void* [n];

		for (int i = 0; i < n; i++)
		{
			args[i] = args_[i];
		}
	}
	void swap_ptr (double** ptr1, double** ptr2)
	{	
		double* temp;	
		temp = (*ptr1);	
		(*ptr1) = (*ptr2);	
		(*ptr2) = temp;
	};



};


