#include "jacobi_iteration_method.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <random>

#define DB(value) static_cast<double>(value)

/* Here we define the solve function for the Jacobi Iteration method, 
   which iterates each value of the local matrix doing an 'average' of
   the values of the adjacent cells. We use A_local alternatively A_local[0]
   and A_local[1] for iterating, where these are vectors of doubles 
   where the matrix local values are kept (A_local[i*n+j] = A[initial row + i][j])

   As this class (JacobiMethod) and friend function may be called with more 
   than one core, A_local represents the part of the matrix managed by the 
   actual core, dividing the number of rows equitably bewteen the cores used 
   in the logical order (core 0 gets the firsts rows)

   Initizalitation of MPI must be done before calling this function
   and must be finalized after using it */

void solve(JacobiMethod &J){
    
  // initializing MPI variables and setting the number od threads

  omp_set_num_threads(2);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // initializing the size of the local matrix (local_n x n) 

  int n = J.n;
  int local_n = (n % mpi_size > mpi_rank) ? n/mpi_size + 1 : n/mpi_size;


  // upper_line and lower_line will be used to get the info of the lines contiguous
  // to A_local from the coniguos cores and f_local will store the function values

  J.A_local.assign(2, std::vector<double>(local_n*n, 0));
  std::vector<double> f_local(local_n*n);
  std::vector<double> upper_line(n, 0);
  std::vector<double> lower_line(n, 0);
  

  // setting the values of the inner values of the local matrix
  int f_row_start = (n/mpi_size)*mpi_rank + std::min(mpi_rank, n%mpi_size);
  for(int i = 1; i < local_n-1; i++){
    for(int j = 1; j < n-1; j++)
      J.A_local[0][i*n+j] = J.fun(DB(f_row_start+i)/n, DB(j)/n); 
  }

  // setting the values of the upper row (if the core is not the first one, core 0)
  if(mpi_rank != 0){
     for(int j = 1; j < n-1; j++)
       J.A_local[0][j] = J.fun(DB(f_row_start)/n, DB(j)/n); 
  }

  //  setting the values of the lower row (if the core is not the last one, core mpi_rank-1)
  if(mpi_rank != mpi_size-1){
     for(int j = 1; j < n-1; j++)
       J.A_local[0][(local_n-1)*n+j] = J.fun(DB(f_row_start+(local_n-1))/n, DB(j)/n); 
  }
  f_local = J.A_local[0];
  

  //  setting the boundary conditions with the dirichlet function 
 
  for(int i = 0; i < local_n; i++){
    J.A_local[0][i*n] = J.dirichlet(DB(f_row_start+i)/(n-1), 0.0);
    J.A_local[0][i*n+n-1] = J.dirichlet(DB(f_row_start+i)/(n-1), 1.0);
  }

  if(mpi_rank == 0){
    for(int j = 0; j < n; j++) 
      J.A_local[0][j] = J.dirichlet(0.0, DB(j)/(n-1));
   }
  
  if(mpi_rank == mpi_size-1){
    for(int j = 0; j < n; j++) 
      J.A_local[0][(local_n-1)*n+j] = J.dirichlet(DB(f_row_start+(local_n-1))/(n-1), DB(j)/(n-1));
  }
  
  J.A_local[1] = J.A_local[0];


  // iterating till convergece or until max_it iterations have been done 
  // it's considered to be converged when for A_local for every core converges

  J.it = 0;
  bool global_convergence = false, local_convergence = false;
  int a = 0, b = 1; //values used to alternate between A_local[0] and A_local[1]
  double h2 = 1.0/pow((n-1),2);

  while(J.it < J.max_it && !global_convergence){
      
    // update each internal entries as the average of the values of a four–point stencil

    #pragma omp parallel for shared(J)
    for(int i = 1; i < local_n-1; i++){
        for(int j = 1; j < n-1; j++)
          J.A_local[b][i*n+j] =  (J.A_local[a][i*n+j+1]   + J.A_local[a][i*n+j-1]  + J.A_local[a][(i+1)*n+j] 
                              + J.A_local[a][(i-1)*n+j] + h2*J.fun(DB(i+f_row_start)/(n-1),DB(j)/(n-1)))/ 4;
    }
    if(mpi_rank < mpi_size-1){
      int i = local_n-1;
      #pragma omp parallel for shared(J, lower_line)
      for(int j = 1; j < n-1; j++)
        J.A_local[b][i*n+j]   =  (J.A_local[a][i*n+j+1]   + J.A_local[a][i*n+j-1] + lower_line[j]       
                              + J.A_local[a][(i-1)*n+j] + h2*J.fun(DB(i+f_row_start)/(n-1),DB(j)/(n-1))) / 4;
    }
    if(mpi_rank > 0){
      #pragma omp parallel for shared(J, upper_line)
      for(int j = 1; j < n-1; j++)
        J.A_local[b][j]       =  (J.A_local[a][j+1]       + J.A_local[a][j-1]      + J.A_local[a][n+j] 
                              + upper_line[j]         + h2*J.fun(DB(f_row_start)/(n-1),DB(j)/(n-1))) / 4;
    }

    // sending the adjacent rows between the cores

    std::vector<MPI_Request> requests(2);  
    std::vector<MPI_Status>  statuses(2);

    if(mpi_rank < mpi_size -1){
      MPI_Isend(J.A_local[b].data()+n*(local_n-1), n, MPI_DOUBLE, mpi_rank+1, mpi_rank, MPI_COMM_WORLD, &requests[0]);
      MPI_Irecv(lower_line.data(), n, MPI_DOUBLE, mpi_rank+1, mpi_rank+1, MPI_COMM_WORLD, &requests[1]);
    }

    if(mpi_rank > 0){
      MPI_Isend(J.A_local[b].data(), n, MPI_DOUBLE, mpi_rank-1, mpi_rank, MPI_COMM_WORLD, &requests[0]);
      MPI_Irecv(upper_line.data(), n, MPI_DOUBLE, mpi_rank-1, mpi_rank-1, MPI_COMM_WORLD, &requests[1]);
    }

    MPI_Waitall(2, requests.data(), statuses.data());

    // compute the convergence criterion as the L2 norm of the increment between J.A_local(k+1) and J.A_local(k)
    // where e2 stands for the square of the error for each core

    double e2 = 0; 
    #pragma omp parallel for reduction(+:e2)
    for(int i = 0; i < local_n; i++){
        for(int j = 1; j < n-1; j++)
            e2 += pow((J.A_local[a][i*n+j]-J.A_local[b][i*n+j]),2);
    } 
    e2 /= n-1;
    local_convergence = (e2 < pow(J.tolerance, 2));
    
    // global_convergence will be true if and only if for all cores local_convergence is true 
    // (MPI_Allreduce with MPI_LAND) 

    MPI_Allreduce(&local_convergence, &global_convergence, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    std::swap(a,b); // we swap between a and b to alternate bewteen A_local[0] and A_local[1]
    ++J.it; 
  }
}