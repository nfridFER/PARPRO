/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

 /* This is an interactive version of cpi */
#include "mpi.h"
#include <cstdio>
#include <iostream>
#include <math.h>


static double f(double a)
{
    return (4.0 / (1.0 + a * a));
}

int main(int argc, char * argv[]) {

	MPI_Init(NULL, NULL);

    unsigned long long done = 0, n, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;
    double startwtime = 0.0, endwtime;

   

    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    while (!done) {
        if (my_id == 0) {
            fprintf(stdout, "Upisite broj elemenata reda: (0 za kraj) ");
            fflush(stdout);
            std::cin >> n;
           
            startwtime = MPI_Wtime();
        }
        MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        if (n == 0)
            done = 1;
        else {
            h = 1.0 / (double)n;
            sum = 0.0;
            for (i = my_id + 1; i <= n; i += world_size) {
                x = h * ((double)i - 0.5);
                sum += f(x);
            }
            mypi = h * sum;
            MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (my_id == 0) {
                printf("Pi je otprilike %.16f, pogreska je %.16f\n",
                    pi, fabs(pi - PI25DT));
                endwtime = MPI_Wtime();
                printf("Proteklo vrijeme = %f\n", endwtime - startwtime);
                fflush(stdout);
            }
        }
    }

	return 0;
}