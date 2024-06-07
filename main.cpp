#include <iostream>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <cmath>
#include "jacobi_iteration_method.hpp"
#include "writeVTK.hpp"

#define DB(value) static_cast<double>(value)

static double c_start, c_diff;
#define tic() c_start = MPI_Wtime();
#define toc(x)                                       \
  {                                                  \
    c_diff = MPI_Wtime() - c_start;                  \
    std::cout << x << c_diff << " [s]" << std::endl; \
  }

double f(double x, double y) {
    double pi = M_PI; // Constant pi
    return 8*std::pow(pi, 2) * std::sin(2 * pi * x) * std::sin(2 * pi * y);
}

double u(double x, double y) {
    double pi = M_PI; // Constant pi
    return std::sin(2 * pi * x) * std::sin(2 * pi * y);
}


int main(int argc, char** argv) {

  
  // Initialize MPI
  int provided; 
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) {
      std::cerr << "Error initializing MPI." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Start counting 
  tic();
 
  // Set the size of the matrix
  int n = pow(2,8);

  // Define the dirichlet function
  std::function<double(double, double)> dirichlet = [](double x, double y) { return 0.0; };
  
  // Initialize the Jacobi Method class
  JacobiMethod J(f, dirichlet, n);

  // Set the tolerance and the max_it
  J.set_tolerance(1e-8);
  J.set_max_it(100000);

  // Call to the solve function
  solve(J);


  // Collecting in a single matrix the information of all the cores 
  
  std::vector<double> result;
  if (mpi_rank == 0) result.resize(n*n);
  
  std::vector<int> recv_counts(mpi_size, 0), recv_start_idx(mpi_size, 0);
  
  int start_idx = 0;

  for (int i = 0; i < mpi_size; ++i)  {
      recv_counts[i] = (n % mpi_size > i) ? n*(n / mpi_size + 1) : n*(n / mpi_size);
      recv_start_idx[i] = start_idx;
      start_idx += recv_counts[i];
  }

  std::vector<double> v = J.A_local[0];
  MPI_Gatherv(v.data(), v.size(), MPI_DOUBLE, result.data(),  recv_counts.data(), 
            recv_start_idx.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
   
  // Finish counting
  if (mpi_rank == 0){
    toc("Jacobi Iteration - Time elapsed on rank " +
          std::to_string(mpi_rank) + ": ");
  }
   
  // Ending MPI
  MPI_Finalize();

  // Calculating the error of the method with the L2 norm 
  if(mpi_rank == 0){  
    double error = 0;
    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++)
        error += pow(result[n*i+j]-u(DB(i)/(n-1),DB(j)/(n-1)),2);

    }
    error /= n;
    error = sqrt(error);
    std::cout<< "It has been done in " << J.get_it() << " iterations.\n";
    std::cout<< "It differs from the actual solution with the L2 norm for " << error << ".\n";
  }

// Printing the VTK file
   if(mpi_rank == 0){  
    std::vector<std:: vector<double>> A;
    for(int i = 0; i < n; i++){
      std::vector<double> row;
      for(int j = 0; j < n; j++)
        row.push_back(result[n*i+j]);
      A.push_back(row);
    }
    generateVTKFile("mesh/out.vtk", A, n-1, n-1, DB(1)/n, DB(1)/n);
  }
    
  return 0;
}
/* RESULTS for the Jacobi method for function f and null dirichlet condition
   for n = 2^k, with tolerance=1e-8 and max_it=100000

k = 4 and core = 2
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 0.00629625 [s]
It has been done in 659 iterations.
It differs from the actual solution with the L2 norm for 0.0276578.

k = 4 and core = 4
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 0.00380825 [s]
It has been done in 661 iterations.
It differs from the actual solution with the L2 norm for 0.0276578.

k = 5 and core = 2
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 0.0302673 [s]
It has been done in 2324 iterations.
It differs from the actual solution with the L2 norm for 0.00939949.

k = 5 and core = 4
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 0.0262934 [s]
It has been done in 2314 iterations.
It differs from the actual solution with the L2 norm for 0.00939949.

k = 6 and core = 2
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 0.314492 [s]
It has been done in 7591 iterations.
It differs from the actual solution with the L2 norm for 0.0032654.

k = 6 and core = 4
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 0.210828 [s]
It has been done in 7523 iterations.
It differs from the actual solution with the L2 norm for 0.0032654.

k = 7 and core = 2
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 3.98046 [s]
It has been done in 22817 iterations.
It differs from the actual solution with the L2 norm for 0.00114588.

k = 7 and core = 4
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 2.71483 [s]
It has been done in 22505 iterations.
It differs from the actual solution with the L2 norm for 0.00114607.

k = 8 and core = 2
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 56.4388 [s]
It has been done in 63751 iterations.
It differs from the actual solution with the L2 norm for 0.000428868.

k = 8 and core = 4
--------------------------------------------------------------------
Jacobi Iteration - Time elapsed on rank 0: 37.517 [s]
It has been done in 62393 iterations.
It differs from the actual solution with the L2 norm for 0.000435491.


We can see that as we use a bigger n the precision of the method gets better.
We can also see that using more core reduces in a notorious way the time. 

*/


