# A matrix–free parallel solver for the Laplace equation

Consider the Laplace equation in [0,1]x[0,1] modelling the heat diffusion over a square domain with a prescribed temperature (Dirichlet
conditions) on the whole boundary.

## Table of Contents

1. [Description](#description)
2. [Instalation and compilation](#instalation-and-compilation)
3. [Usage](#usage)
4. [License](#license)
5. [Contact](#contact)

## Description

A possible approach to solve this problem is the so called Jacobi iteration method: given a uniform Cartesian decomposition of Ω consisting of n points along each coordinate direction, the goal is to find the discrete solution uij = u(xi
, yj ), i, j = 1, . . . , n at each point of such
Cartesian grid.

We aim at representing the solution as a (dense) matrix U of size n × n; the matrix is
initialized with zeroes, except for the first and last rows and columns, which contain the
boundary condition values defined in eq.

The files consist on main.cpp, jacobi_iterattion_method.cpp, jacobi_iterattion_method.hpp and a Makefile.
Also the writeVTK.hpp is used to print the resulting matrix in VTK format and the out.vtk file is an example of it.

In the files of the repository we define a JacobiMethod class with a friend function which solves the Laplace equation. 

The code is thought to be executed in several cores, and MPI functions are used to communicate data between adjacent processors.
In the code we also use OpenMP directive to further parallelize the local computations.

## Instalation and compilation

### Prerequisites

Ensure you have the following installed:
- MPI implementation (e.g., OpenMPI or MPICH)
- C++ compiler that supports C++17
- OpenMP support

### Steps

1. **Clone the repository**

   ```sh
   git clone git@github.com:miguelalcaniz/Challenge3_PACS.git
   cd Challenge3_PACS

2. **Compile the code**
   ```sh
   make
3. **Execute the code using 'k' processors**
   ```sh
   mpiexec -np k main
## Usage 

First #include "jacobi_iteration_method.hpp".
As shown in the main.cpp file, first we initialize MPI.
Then we create a JacobiMethod object giving the f, the dirichlet function (use the one in main.cpp i it is all 0) and n, the size of the nxn matrix representing the [0,1]x[0,1] domain.
If you want to change the max_it or the tolerance use the set_tolerance() and set_max_it() methods of the class.
After, call the solve function with the Jacobi method object as argument.
Finally collect all the processors information and finish MPI as shown in the main.cpp.

### License

This project is licensed under the MIT License - see the [LICENSE] file for details.

### Contact

If you have any questions, suggestions, or issues, feel free to contact me:

*Email:* miguelalcaniz02@gmail.com
