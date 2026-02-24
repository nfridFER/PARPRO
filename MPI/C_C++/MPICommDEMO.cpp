#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
	
	MPI_Init(NULL, NULL);
	
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Potrebna najmanje 2 procesa
	if (world_size < 2) {
		fprintf(stderr, "Mora biti najmanje 2 procesa %s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	int number;

	//slanje u lancu, proc id=0 počinje, zatim čeka što će dobiti od n-1-og
	if (world_rank == 0) {
		number = 0;
		MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		printf("Proces %d salje broj %d procesu %d\n", world_rank, number, world_rank + 1);
		MPI_Recv(&number, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Proces %d prima broj %d od procesa %d\n", world_rank, number, world_size - 1);
	}
	else{
		MPI_Recv(&number, 1, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Proces %d prima broj %d od procesa %d\n", world_rank,number, world_rank-1);
		number++;
		MPI_Send(&number, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
		printf("Proces %d salje broj %d procesu %d\n", world_rank, number, (world_rank + 1) % world_size);
	}
	MPI_Finalize();


	return 0;
}


