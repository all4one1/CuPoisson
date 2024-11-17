## Clone the project 

```
git clone --recurse-submodules https://github.com/all4one1/CuPoisson
```

may be needed:
```
git submodule update --remote --recursive
```


## Usage:
```cpp
#include "CuPoisson.h"

__global__ void poisson_kernel(double* p, double* p0, double* rhs, double* extra)
{
	// some stuff
}


int main()
{
	int N = 10000; 
	
	dim3 blockDim = 512;
	dim3 gridDim = (N + blockDim.x - 1) / blockDim.x;
	
	double* p, * p0, * rhs, *some_extra_field; //device pointers

	CuPoisson poisson;

	void* args[] = { &p, &p0, &rhs, &some_extra_field };
	poisson.set_kernel(poisson_kernel, args, gridDim, blockDim);
	poisson.set_main_field(p, p0, N);

	return 0;
}
