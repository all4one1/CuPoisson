#pragma once
#include "CuPoisson.h"


//__global__ void print_constants()
//{
//	printf("Address: %p \n", (void*)&dev);
//	printf("hx, hy, hz = %f, %f, %f \n", dev.hx, dev.hy, dev.hz);
//	printf("nx, ny, nz = %i, %i, %i \n", dev.nx, dev.ny, dev.nz);
//	printf("offset, offset2 = %i, %i \n", dev.offset, dev.offset2);
//	printf("tau_p = %f \n", dev.tau_p);
//	printf("PoissonState: %i \n", static_cast<int>(dev.boundary));
//}



CuPoisson::CuPoisson()
{
	//N = 
	//Nbytes = N * sizeof(double);
	//cuRed = CudaReduction(f_dev, N, 1024);
}



//void CuPoisson::solve()
//{
//	{
//		eps = 1.0;
//		res0 = 0.0;
//		res = 0.0;
//		k = 0;
//
//		//while (eps > eps0 * res0 || k < 2)
//		while (eps > eps0 * res0)
//		{
//
//			res = 0.0;
//			k++;
//
//			Poisson << < gridDim, blockDim >> > (f_dev, f0_dev, ux_dev, uy_dev, uz_dev, tau);
//
//
//
//			auto swap_ptr = [](double** ptr1, double** ptr2)
//			{	double* temp;	temp = (*ptr1);	(*ptr1) = (*ptr2);	(*ptr2) = temp; };
//
//			swap_ptr(&f_dev, &f0_dev);
//
//
//			////swap_one << < ceil(size_l / 1024.0), 1024 >> > (dev0, dev);
//
//
//			eps = abs(res - res0);
//			res0 = res;
//
//			if (k % 1000 == 0)
//			{
//				//cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
//				//cout << k << "  " << setprecision(15) << p_h[1 + off + off2] - p_h[nx - 1 + off * (ny - 1) + off2 * (nz - 1)] << " " << eps << endl;
//			}
//
//		}
//	}
//}

//void CuPoisson::copy_to_host(double* host_ptr)
//{
//	//cudaMemcpy(host_ptr, f_dev, Nbytes, cudaMemcpyDeviceToHost);
//}
