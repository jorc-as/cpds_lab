
#include "omp.h"

#define lowerb(id, p, n)  ( id * (n/p) + (id < (n%p) ? id : n%p) )
#define numElem(id, p, n) ( (n/p) + (id < (n%p)) )
#define upperb(id, p, n)  ( lowerb(id, p, n) + numElem(id, p, n) - 1 )

#define min(a, b) ( (a < b) ? a : b )
#define max(a, b) ( (a > b) ? a : b )

extern int userparam;

// Function to copy one matrix into another (excluding boundary cells)
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey){
	#pragma omp parallel
	{
		int n_blocks = omp_get_num_threads();
		int block_id = omp_get_thread_num();
		int i_start = lowerb(block_id, n_blocks, sizex);
		int i_end = upperb(block_id, n_blocks, sizex);

    		for (int i = max(1, i_start); i <= min(sizex - 2,i_end) ; i++) {
    		    for (int j = 1; j < sizey - 1; j++) {
    		        v[i * sizey + j] = u[i * sizey + j];
    		    }
    		}
	}
}

double solve_gauss (double *u, unsigned sizex, unsigned sizey);

// 2D solver: one iteration step (Jacobi)
double solve (double *u, double *unew, unsigned sizex, unsigned sizey) {
    double tmp, diff, sum = 0.0;

    if (u == unew) return solve_gauss(u, sizex, sizey);

    #pragma omp parallel private(diff, tmp) reduction(+ : sum)   // complete data sharing constructs here
    {
        int nthreads = omp_get_num_threads();

        int my_id = omp_get_thread_num();

	int aux_size = sizex - 2;	// I subtract 2 to sizex because we don't want to account for the first and last rows
        				// , since they don't have rows above and below respectively, but we also have to add 1
        				// to the total value, since the macros start indexing from the first row.
					// Also, you can't put the expression directly inside the macro call because 
					// the preprocessor substitutes arguments textually. With n replaced by sizex-2, 
					// the expression n/p becomes sizex-2/nthreads, which parses as sizex - (2/nthreads) 
					// which is incorrect.

        int i_start = lowerb(my_id, nthreads, aux_size) + 1;
        int i_end = upperb(my_id, nthreads, aux_size) + 1;

        for (int i = i_start; i <= i_end; i++) {
            for (int j = 1; j < sizey - 1; j++) {
                tmp = 0.25 * ( u[ i * sizey       + (j - 1) ] +   // left
                               u[ i * sizey       + (j + 1) ] +   // right
                               u[ (i - 1) * sizey + j       ] +   // top
                               u[ (i + 1) * sizey + j       ] );  // bottom
                diff = tmp - u[i * sizey + j];
                sum += diff * diff;
                unew[i * sizey + j] = tmp;
            }
        }
    }

    return sum;
}
/* Blocked code with more loop which imply more overhead:

// Function to copy one matrix into another
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey) {

    int nblocksi=1;
    int nblocksj=1;

    for (int blocki=0; blocki<nblocksi; ++blocki) {
      int i_start = lowerb(blocki, nblocksi, sizex);
      int i_end = upperb(blocki, nblocksi, sizex);
      for (int blockj=0; blockj<nblocksj; ++blockj) {
        int j_start = lowerb(blockj, nblocksj, sizey);
        int j_end = upperb(blockj, nblocksj, sizey);
        for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++)
          for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++)
            v[i*sizey+j] = u[i*sizey+j];
      }
    }
}
*/

/* Jacobi Blocked code: 
 *
double solve (double *u, double *unew, unsigned sizex, unsigned sizey) {
    double tmp, diff, sum=0.0;

    if (u==unew) return solve_gauss(u,sizex,sizey);

    int nblocksi=omp_get_max_threads();
    int nblocksj=1;

    #pragma omp parallel     // complete data sharing constructs here
    {
      int blocki = ... ;
      int i_start = lowerb(blocki, nblocksi, sizex);
      int i_end = upperb(blocki, nblocksi, sizex);
      for (int blockj=0; blockj<nblocksj; ++blockj) {
        int j_start = lowerb(blockj, nblocksj, sizey);
        int j_end = upperb(blockj, nblocksj, sizey);
        for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
          for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++) {
	        tmp = 0.25 * ( u[ i*sizey	   + (j-1) ] +  // left
                           u[ i*sizey	   + (j+1) ] +  // right
                           u[ (i-1)*sizey + j     ] +  // top
                           u[ (i+1)*sizey + j     ] ); // bottom
	        diff = tmp - u[i*sizey+ j];
	        sum += diff * diff;
	        unew[i*sizey+j] = tmp;
          }
        }
      }
    }

    return sum;
}
*/

// 2D-blocked solver: one iteration step
double solve_gauss (double *u, unsigned sizex, unsigned sizey) {
    double tmp, diff, sum=0.0;

    int nblocksi=omp_get_max_threads();
    int nblocksj=nblocksi; /* Change to a proper value, since 1 implies no blocking. */

	#pragma omp parallel for ordered(2) private(tmp, diff) reduction(+: sum) // complete data sharing constructs here
      
		for (int blocki=0; blocki<nblocksi; ++blocki) {
			/* In a block decomposition by rows each thread 
			 * will take one iteration of this outer loop. */
		     
		  int i_start = lowerb(blocki, nblocksi, sizex);
		  int i_end = upperb(blocki, nblocksi, sizex);
		  for (int blockj=0; blockj<nblocksj; ++blockj) {
		    int j_start = lowerb(blockj, nblocksj, sizey);
		    int j_end = upperb(blockj, nblocksj, sizey);

		    #pragma omp ordered \
		    depend(sink: blocki, blockj-1) \
		    depend(sink: blocki-1, blockj)
			// Update current tile
			for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
			  for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++) {
				tmp = 0.25 * ( u[ i*sizey	   + (j-1) ] +  // left
					   u[ i*sizey	   + (j+1) ] +  // right
					   u[ (i-1)*sizey + j     ] +  // top
					   u[ (i+1)*sizey + j     ] ); // bottom
				diff = tmp - u[i*sizey+ j];
				sum += diff * diff;
				u[i*sizey+j] = tmp;
			  }
			}
		    #pragma omp ordered depend(source)
		  }
		}

    return sum;
}
