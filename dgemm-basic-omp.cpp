#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"
#include <cmath>
#include <vector>
#include <cstring>


const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C) 
{
   // insert your code here: implementation of basic matrix multiply with OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

#pragma omp parallel
   {
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME)
#pragma omp for collapse(2)
      for(int x = 0; x < n ; x++) {
         for(int y = 0; y < n; y++) {
            double square_value = C[x*n + y];

            for(int z = 0; z < n; z++) {
               square_value += A[x*n + z] * B[z*n + y];
            }

            C[x*n + y] = square_value;
         }
      }
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   }

}
