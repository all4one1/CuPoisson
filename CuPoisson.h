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


