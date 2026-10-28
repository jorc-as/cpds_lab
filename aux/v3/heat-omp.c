/*
 * Iterative solver for heat distribution
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "heat.h"
#include "fake.h"

int userparam=1; // global variable to store extra parameter received as argument

void usage( char *s ) {
    fprintf(stderr, "Usage: %s input_file [-n num_iter -s resolution -a solver -u extra -o result_file]\n", s);
    fprintf(stderr, "       input_file with heat sources\n");
    fprintf(stderr, "       -n to specify the number of iterations to stop iterative algorithm (0 no stop, 25000 default)\n");
    fprintf(stderr, "       -s to specify the resolution (size) of the squared hot surface (254 default)\n");
    fprintf(stderr, "       -r to specify the residual value to stop iterative algorithm (0.00005 default)\n");
    fprintf(stderr, "       -a to specify the solver to use. 0: Jacobi; 1: Gauss-Seidel (default 0)\n");
    fprintf(stderr, "       -o to specify the name of the output file with heat map generated (default heat.ppm)\n");
    fprintf(stderr, "       -u extra argument to be used by the programmer\n");
}

int main( int argc, char *argv[] ) {
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;

#ifdef _OPENMP
    TRACE_START_MARK;
#endif

    // algorithmic parameters
    algoparam_t param;
    int np;

    double runtime, flop;
    double residual=0.0;

    // Set defaults for algorithmic parameters
    param.maxiter = 25000;            // -n argument
    param.resolution = 254;           // -s argument
    param.residual = 0.00005;         // -r argument
    param.visres = param.resolution;
    param.algorithm = 0;              // -a argument, Jacobi as default

    // check arguments
    if( argc < 2 ) {
	    usage(argv[0]);
	    return EXIT_FAILURE;
    }

    // check input file
    if( !(infile=fopen(argv[1], "r"))  ) {
	    fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
	    usage(argv[0]);
	    return EXIT_FAILURE;
    }

    resfilename = "heat.ppm";
    // Process optional command-line arguments
    for (int i=2; i<argc-1; i++) {
        if (strcmp(argv[i], "-n")==0) {
            param.maxiter = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-s")==0) {
            param.resolution = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-r")==0) {
            param.residual = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-a")==0) {
            param.algorithm = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-u")==0) {
            userparam = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-o")==0) {
            resfilename = argv[++i];
        }
        else {
            return EXIT_FAILURE;
        }
     }

     // check result file
    if (!(resfile=fopen(resfilename, "w")) ) {
	    fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n", resfilename);
	    usage(argv[0]);
	    return EXIT_FAILURE;
    }

    // check input file with heat sources
    if( !read_input(infile, &param) ) {
        fprintf(stderr, "\nError: Error parsing input file.\n");
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    print_params(&param);

    if( !initialize(&param) ) {
	    fprintf(stderr, "Error in Solver initialization.\n\n");
	    usage(argv[0]);
        return EXIT_FAILURE;
	}

    // full size (param.resolution are only the inner points)
    np = param.resolution + 2;

    // starting time
    runtime = wtime();

    iter = 0;
    while(1) {
	switch( param.algorithm ) {
	    case 0: // JACOBI
	            residual = solve(param.u, param.uhelp, np, np);
		    double* aux = param.u;
		    param.u = param.uhelp;
		    param.uhelp = aux;
		    // Copy uhelp into u
		    //copy_mat(param.uhelp, param.u, np, np);
		    break;
	    case 1: // GAUSS-SEIDEL
                    residual = solve(param.u, param.u, np, np);
		    break;
        default: // WRONG OPTION
            fprintf(stdout, "Error: solver not implemented, exiting execution \n");
            return EXIT_FAILURE;
	    }

        iter++;

        // solution good enough ?
        if (residual < param.residual) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (param.maxiter>0 && iter>=param.maxiter) break;
    }

    // Flop count after iter iterations
    flop = iter * 11.0 * param.resolution * param.resolution;
    // stopping time
    runtime = wtime() - runtime;

    fprintf(stdout, "Time: %04.3f \n", runtime);
    fprintf(stdout, "Flops and Flops per second: (%3.3f GFlop => %6.2f MFlop/s)\n",
	    flop/1000000000.0,
	    flop/runtime/1000000);
    fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

    // for plot...
    coarsen( param.u, np, np,
	     param.uvis, param.visres+2, param.visres+2 );

    write_image( resfile, param.uvis,
		 param.visres+2,
		 param.visres+2 );

    finalize( &param );

#ifdef _OPENMP
    TRACE_END_MARK;
#endif

    return 0;
}
