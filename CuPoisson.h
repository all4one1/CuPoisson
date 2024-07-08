#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#include "CudaReduction/CuReduction.h"



struct CuPoisson
{
	unsigned int N = 0, Nbytes = 0;
	unsigned int k = 0;
	double eps = 0, res = 0, res0 = 0;
	double eps0 = 1e-5;

	CudaReduction cuRed;
	dim3 gridDim, blockDim, gridLinear;


	CuPoisson();

	void solve(double *f_dev, double* f0_dev, double* ux_dev, double* uy_dev, double* uz_dev, 
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

}


