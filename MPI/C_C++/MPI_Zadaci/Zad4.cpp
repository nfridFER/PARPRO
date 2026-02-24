#include <mpi.h>
#include <stdio.h>

int main(int argc, char * argv[]) {

	MPI_Init(NULL, NULL);

	int ID;
	MPI_Comm_rank(MPI_COMM_WORLD, &ID);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int a=0, b=0, c=0;

	if (ID == 1) {
		MPI_Send(&ID, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
		MPI_Recv(&a, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&a, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
		MPI_Recv(&b, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		c = 2 * a + b;
	}
	else if (ID == 2) {
		MPI_Recv(&a, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&b, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		c = 2 * a + b;
		MPI_Send(&c, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&ID, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
	}
	else if (ID == 3) {
		MPI_Send(&ID, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
		MPI_Recv(&a, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&b, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		c = 2 * a + b;
	}

	printf("Proces %d, c=%d", ID, c);

	MPI_Finalize();

	return 0;
}


/* Mogući rezultati
c	1	2	3

	16	7	11
	15	5	12
	12	5	9
	21	7	16
	
*/