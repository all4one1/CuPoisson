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


__global__ void swap_(double* f_old, double* f_new, unsigned int n) 
{
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < n)	f_old[l] = f_new[l];
}


struct CuPoisson
{
	double eps_iter = 1e-5;
private:
	unsigned int N = 0, k = 0;
	double eps = 0, res = 0, res0 = 0;
	CudaReduction* CR = nullptr;
	dim3 gridDim, blockDim;
	size_t smem = 0, Nbytes = 0;
	cudaStream_t stream = 0;
	void* kernel = nullptr;
	void** args = nullptr;
	double* f_dev = nullptr, * f0_dev = nullptr;
	std::pair<int, int> main_index;

	bool logs_out = false;
	std::ofstream k_write;

private:
	double* ptr_(int i)
	{
		return *(double**)(args[i]);
	}

public:
	CuPoisson() {}
	CuPoisson(unsigned int N_, void* kernel_, void** args_, std::pair<int, int> main, dim3 grid_, dim3 block_, size_t smem_ = 0, cudaStream_t stream_ = 0)
	{
		set_kernel(N_, kernel_, args_, main, grid_, block_, smem_ = 0, stream_ = 0);
	}

	/**
	* N - the size of
	* kernel - __global__ void kernel function
	* args - all the arguments of the kernel
	* main - {i1, i2}, indices of main fields to swap and to compute reduction
	* grid (n of blocks)
	* block (n of threads per block)
	* shared memory and stream are null by default
	*/
	void set_kernel(unsigned int N_, void* kernel_, void** args_, std::pair<int, int> main, dim3 grid_, dim3 block_, size_t smem_ = 0, cudaStream_t stream_ = 0)
	{
		gridDim = grid_;
		blockDim = block_;
		kernel = kernel_;
		args = args_;
		smem = smem_;
		stream = stream_;

		main_index = main;
		N = N_;		
		Nbytes = sizeof(double) * N;
		CR = new CudaReduction(ptr_(main_index.first), N, 512);
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
			res = CR->reduce(ptr_(main_index.first), true);
			eps = abs(res - res0) / (res0 + 1e-5);
			res0 = res;

			std::swap(args[main_index.first], args[main_index.second]);

			if (eps < eps_iter)	break;
			if (k % 100 == 0) std::cout << "device k = " << k << ", eps = " << eps << std::endl;
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
				std::swap(args[main_index.first], args[main_index.second]);


				res = CR->reduce(ptr_(main_index.second), true);
				eps = abs(res - res0) / (res0 + 1e-5);
				res0 = res;

				if (eps < eps_iter)		break;
			}
			else
			{
				cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
				std::swap(args[main_index.first], args[main_index.second]);


				if (k % k_frequency == 0)
				{
					res0 = CR->reduce(ptr_(main_index.second));

					cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
					std::swap(args[main_index.first], args[main_index.second]);


					k++;
					res = CR->reduce(ptr_(main_index.second));

					eps = abs(res - res0) / (res0 + 1e-5);

					if (eps < eps_iter)		break;
				}
			}
		}
		if (k > 100) std::cout << "device k = " << k << ", eps = " << eps << std::endl;
		if (logs_out) k_write << k << " " << res << " " << eps << std::endl;

	}


	void switch_writting(std::string name_)
	{
		logs_out = true;
		k_write.open(name_ + ".dat");
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