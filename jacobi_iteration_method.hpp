#ifndef JACOBI_ITERATION_METHOD_HPP
#define JACOBI_ITERATION_METHOD_HPP

#include <iostream>
#include<functional>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <random>


class JacobiMethod {
public:

  /// Default constructor.
  JacobiMethod(const std::function<double(double, double)> & f,
               const std::function<double(double, double)> & fdirichlet, 
               const size_t n_) 
    : n(n_) // size of the nxn matrix
    , fun(f) // function for the JacobiMethod
    , dirichlet(fdirichlet) // function for the boundary condition
    , A_local() // vector where the matrix values are stored
    , max_it(40000) // max number of iterations, if not changed settled to 40000
    , tolerance(10e-6) // tolerance for the increment of the iterations, if not changed settled to 10e-6
  {}

  // Method to change the tolerance
  void set_tolerance(const double t){
    tolerance = t;
  }

  // Method to change the maximum number of iterations
  void set_max_it(const unsigned int m){
    max_it = m;
  }

  // sSolving method defined in the jacobi_interation_method.cpp file
  friend void solve(JacobiMethod &J);

  // Method that gives you the iterations used in the solve function
  unsigned int get_it() const{
    return it;
  }
  
  // part of the matrix handled by the actual core
  std::vector<std::vector<double>> A_local;

private:

  // number of rows of the entire matrix and the local one (for this core) 
  const size_t n;
  size_t local_n; 


  // number of max iterations
  unsigned int max_it;

  // iterations done
  unsigned int it;

  // tolerance of the increment
  double tolerance;

  // function f
  const std::function<double(double, double)> fun;

  // function dirichlet for the boundary condition
  const std::function<double(double, double)> dirichlet;

};


#endif /* JACOBI_ITERATION_METHOD_HPP */