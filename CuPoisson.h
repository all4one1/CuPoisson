#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <utility>

#include "CudaReduction/CuReduction.h"





struct CuPoisson
{
	unsigned int N = 0, k = 0;
	double eps = 0, res = 0, res0 = 0;
	double eps_iter = 1e-4;
	CudaReduction* CR = nullptr;
	dim3 gridDim, blockDim;
	size_t smem = 0, Nbytes = 0;
	cudaStream_t stream = 0;
	void* kernel = nullptr;
	void** args = nullptr;
	double* f_dev = nullptr, * f0_dev = nullptr;

	bool logs_out = false;
	std::ofstream k_write;

	/**
	* main field (variable) to calculate and swap
	* N - the size of
	*/
	void set_main_field(double* f_, double* f0_, unsigned int N_)
	{
		f_dev = f_;
		f0_dev = f0_;
		N = N_;
		Nbytes = sizeof(double) * N;
		CR = new CudaReduction(f_dev, N, 512);
	}

	/**
	* kernel - __global__ void kernel function
	* args - all the arguments of the kernel
	* grid (n of blocks)
	* block (n of threads per block)
	* shared memory and stream are null by default
	*/
	void set_kernel(void* kernel_, void** args_, dim3 grid_, dim3 block_, size_t smem_ = 0, cudaStream_t stream_ = 0)
	{
		gridDim = grid_;
		blockDim = block_;
		kernel = kernel_;
		args = args_;
		smem = smem_;
		stream = stream_;
	}

	void switch_writting(std::string name_)
	{
		logs_out = true;
		k_write.open(name_ + ".dat");
	}

	void solve()
	{
		k = 0;
		eps = 1.0;
		res = 0.0;
		res0 = 0.0;


		for (k = 1; k < 1000000; k++)
		{
			cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
			res = CR->reduce(f_dev, true);
			eps = abs(res - res0) / (res0 + 1e-5);
			res0 = res;


			std::swap(args[0], args[1]);
			std::swap(f_dev, f0_dev);

			if (eps < eps_iter)	break;
			if (k % 1000 == 0) std::cout << "device k = " << k << ", eps = " << eps << std::endl;
		}
		if (k > 100) std::cout << "device k = " << k << ", eps = " << eps << std::endl;
		if (logs_out) k_write << k << " " << res << " " << eps << std::endl;
	}

	void solve_v2(unsigned int k_minimal_threshold, unsigned int k_frequency)
	{
		k = 0;
		eps = 1.0;
		res = 0.0;
		res0 = 0.0;


		for (k = 1; k < 1000000; k++)
		{
			if (k % 1000 == 0) std::cout << "device k = " << k << ", eps = " << eps << std::endl;

			if (k < k_minimal_threshold)
			{
				cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
				std::swap(args[0], args[1]);
				std::swap(f_dev, f0_dev);


				res = CR->reduce(f0_dev, true);
				eps = abs(res - res0) / (res0 + 1e-5);
				res0 = res;

				if (eps < eps_iter)		break;
			}
			else
			{
				cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
				std::swap(args[0], args[1]);
				std::swap(f_dev, f0_dev);

				if (k % k_frequency == 0)
				{
					res0 = CR->reduce(f0_dev);

					cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
					std::swap(args[0], args[1]);
					std::swap(f_dev, f0_dev);

					k++;
					res = CR->reduce(f0_dev);

					eps = abs(res - res0) / (res0 + 1e-5);

					if (eps < eps_iter)		break;
				}
			}
		}
		if (k > 100) std::cout << "device k = " << k << ", eps = " << eps << std::endl;
		if (logs_out) k_write << k << " " << res << " " << eps << std::endl;

	}
};

//swap_one_3<<<gridDim, blockDim>>>(f0_dev, f_dev, N);

//__global__ void swap_one_3(double* f_old, double* f_new, unsigned int N)
//{
//	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
//	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
//	unsigned int l = i + dev.offset * j + dev.offset2 * k;
//	if (l < N)	f_old[l] = f_new[l];
//}