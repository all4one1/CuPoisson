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
	double eps0 = 1e-5;
	CudaReduction CR;
	dim3 gridDim, blockDim;
	size_t smem = 0, Nbytes = 0;
	cudaStream_t stream = 0;
	void* kernel = nullptr;
	void** args = nullptr;


	CuPoisson(unsigned int grid_, unsigned int block_, unsigned int N_, void* kernel_, std::vector<void*> args_)
	{
		gridDim = grid_;
		blockDim = block_;
		N = N_;
		Nbytes = sizeof(double) * N;

		set_kernel_and_args(kernel_, args_);
	}


	void solve()
	{
		CudaReduction CR(N, 1024);
		k = 0;
		eps = 1.0;
		res = 0.0;
		res0 = 0.0;

		for (k = 0; k < 200000; k++)
		{

			kernelSolveJacobi << < kernel.Grid1D, kernel.Block1D >> > (f, f0, b, N, M);
			swap_one << < kernel.Grid1D, kernel.Block1D >> > (f0, f, N);

			res = CR.reduce();
			eps = abs(res - res0) / res0;
			res0 = res;

			if (eps < eps_iter)
			{
				break;
			}

			if (k % 1000 == 0) cout << "device k = " << k << ", eps = " << eps << endl;

		}
		cout << "device k = " << k << ", eps = " << eps << endl;
	}


	void set_kernel_and_args(void* kernel_, std::vector<void*> args_)
	{
		kernel = kernel_;
		size_t N = args_.size();
		args = new void* [N];

		for (int i = 0; i < N; i++)
		{
			args[i] = args_[i];
		}
	}

	cudaError_t run_kernel()
	{
		return cudaLaunchKernel(kernel, gridDim, blockDim, args, smem, stream);
	}

	void solve(double* f_dev, double* f0_dev, double* ux_dev, double* uy_dev, double* uz_dev,
		void (*kernel)(double*, double*, double*, double*, double*))
	{
		eps = 1.0;
		res0 = 0.0;
		res = 0.0;
		k = 0;

		//while (eps > eps0 * res0 || k < 2)
		while (eps > eps0 * res0)
		{

			res = 0.0;
			k++;

			kernel << <gridDim, blockDim >> > (f_dev, f0_dev, ux_dev, uy_dev, uz_dev);


			auto swap_ptr = [](double** ptr1, double** ptr2)
			{	double* temp;	temp = (*ptr1);	(*ptr1) = (*ptr2);	(*ptr2) = temp; };

			swap_ptr(&f_dev, &f0_dev);
			//swap_one << < ceil(size_l / 1024.0), 1024 >> > (dev0, dev);


			eps = abs(res - res0);
			res0 = res;

			if (k % 1000 == 0)
			{
				//cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
				//cout << k << "  " << setprecision(15) << p_h[1 + off + off2] - p_h[nx - 1 + off * (ny - 1) + off2 * (nz - 1)] << " " << eps << endl;
			}

		}
	};

};


