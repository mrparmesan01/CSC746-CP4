#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";

void square_dgemm(int n, double* A, double* B, double* C) {
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
         double square_sum = C[i*n + j];
         for (int k = 0; k < n; k++) {
            square_sum += A[i*n + k] * B[k*n + j];
         }
         C[i*n + j] = square_sum;
      }
   }
}

/* This routine copies a block from src to dest */
void copy(double *dest, double *src, int n, int block_size) {
   for (int y = 0; y < block_size; y++) {
      std::memcpy(&dest[y * block_size],
         &src[y * n],
         block_size * sizeof(double));
   }
}

/*
* This routine writes a block from src to dest
* where dest is a larger matrix and src is a smaller block
* dest is the larger matrix, src is the smaller block
*/
void write(double *dest, double *src, int n, int block_size) {
   for (int y = 0; y < block_size; y++) {
      // copy the block from src to dest
      // dest[y * n] is the starting point of the row in the larger matrix
      // src[y * block_size] is the starting point of the row in the smaller block
      std::memcpy(&dest[y * n],
         &src[y * block_size],
         block_size * sizeof(double));
   }
}



/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   // std::cout << "Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here " << std::endl;

   const int total_block_size = n / block_size;
   const int arr_size = total_block_size * block_size;

#pragma omp parallel
   {
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);

      // Allocate thread-local variables to be computed in-parallel.
      std::vector<double> buffer(arr_size * 3);
      double* A = buffer.data();
      double* B = buffer.data() + arr_size;
      double* C = buffer.data() + (arr_size * 2);

#pragma omp for collapse(2)
      for(int x = 0; x < total_block_size; x++) {
         for(int y = 0; y < total_block_size; y++) {
            
            // Compute pointer to current block within matrix C.
            double* c_arr_value = &C[x * n * total_block_size + y * block_size];

            // Loading destination block from matrix C to local working memory
            copy(C, c_arr_value, n, block_size);

            // Accumulate across inner block-row/col dimensions
            for(int z = 0; z < n; z++) {
               copy(A, &A[x * n * block_size + z * block_size], n, block_size);
               copy(B, &B[z * n * block_size + y * block_size], n, block_size);

               // Computing matrix multiplication in CPU-cache (L1/L2) memory.
               square_dgemm(block_size, A, B, C);
            }

            // Writes the accumulated result from local memory back to global memory.
            write(c_arr_value, C, n, block_size);
         }
      }

      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   }
}
