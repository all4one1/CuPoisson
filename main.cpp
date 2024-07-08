#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

using namespace std;


struct Constants
{
	double hx, hy, hz, Lx, Ly, Lz, Volume;
	double tau, tau_p;
	double sinA, cosA, sinB, cosB, sinG, cosG;
	double vibr_x, vibr_y, vibr_z;
	double grav_x, grav_y, grav_z;
	double density_x, density_y, density_z;
	double Sc11, Sc12, Sc21, Sc22, psi1, psi2, psiS, Q, Ra, Rav, K, Pr;
	unsigned int nx, ny, nz, N, offset, offset2;
	unsigned int border_type, border_type_heat;
};

struct CudaLaunchSetup
{
	dim3 Grid3D, Block3D, Grid1D, Block1D;
	unsigned int thread_x = 8, thread_y = 8, thread_z = 8, thread_1D = 1024;

	CudaLaunchSetup(unsigned int nx, unsigned int ny, unsigned nz, unsigned int N)
	{
		Grid3D = dim3(
			(unsigned int)ceil((nx + 1.0) / thread_x),
			(unsigned int)ceil((ny + 1.0) / thread_y),
			(unsigned int)ceil((nz + 1.0) / thread_z));
		Block3D = dim3(thread_x, thread_y, thread_z);

		Grid1D = dim3((unsigned int)ceil((N + 0.0) / thread_1D));
		Block1D = thread_1D;
	};
};

__constant__ Constants dev;



__device__  double dx1(unsigned int l, double* f) {
	return 0.5 * (f[l + 1] - f[l - 1]) / dev.hx;
}
__device__  double dy1(unsigned int l, double* f) {
	return 0.5 * (f[l + dev.offset] - f[l - dev.offset]) / dev.hy;
}
__device__  double dz1(unsigned int l, double* f) {
	return 0.5 * (f[l + dev.offset2] - f[l - dev.offset2]) / dev.hz;
}


__global__ void PoissonKernel(double* F, double *F0)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
	unsigned int l = i + dev.offset * j + dev.offset2 * k;


	if (i <= dev.nx && j <= dev.ny && k <= dev.nz && l < dev.N)
	{
		if (i > 0 && i < dev.nx && j > 0 && j < dev.ny && k > 0 && k < dev.nz)
		{
			double rhs = 0; /*  your rhs here  */
			F[l] = F0[l] + dev.tau_p * ((dx1(l, F0) + dy1(l, F0) + dz1(l, F0)) + rhs);
		}
		else if (j == 0)
		{
			F[l] = 0.0;
		}
		else if (j == dev.ny)
		{
			F[l] = 0.0;
		}
		else if (i == 0 && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			F[l] = 0.0;
		}
		else if (i == dev.nx && (j > 0 && j < dev.ny && k > 0 && k < dev.nz))
		{
			F[l] = 0.0;
		}
		else if (k == 0 && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			F[l] = 0.0;
		}
		else if (k == dev.nz && (i > 0 && i < dev.nx && j > 0 && j < dev.ny))
		{
			F[l] = 0.0;
		}
		else
		{
			F[l] = 0.0;
		}
	}
}



#include "CuPoisson.h"

int main()
{



	return 0;
}