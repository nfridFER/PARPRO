#include <mpi.h>
#include <stdio.h>


int main(int argc, char* argv[]) {

	// inicijalizacija
	MPI_Init(NULL, NULL);

	// ukupan broj procesa
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// moj redni broj
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// ime procesora
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	// hello world poruka
	printf("Hello world from processor %s, rank %d"
		" out of %d processors\n",
		processor_name, world_rank, world_size);

	// kraj programa
	MPI_Finalize();

	return 0;
}
