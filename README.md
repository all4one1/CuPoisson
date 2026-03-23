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

	void* args[] = { &p, &p0, &rhs, &some_extra_field };
	CuPoisson poisson((N, poisson_kernel, args, gridDim, blockDim);

	poisson.solve();

	return 0;
}
