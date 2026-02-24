#include <mpi.h>
#include <stdio.h>

int main(int argc, char * argv[]) {

	MPI_Init(NULL, NULL);


	int my_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Potrebna najmanje 2 procesa
	if (world_size < 2) {
		fprintf(stderr, "Mora biti najmanje 2 procesa!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}


	//svaki proces ima jedan element liste (pretpostavit cemo da mu je vrijednost jednaka ID-ju)
	double my_element = (double) my_id;
	int my_left = (my_id + world_size-1) % world_size;
	int my_right = (my_id+1)%world_size;
	
	double left_val;
	double right_val;
		
	
	// parni prvo salju, neparni primaju pa obrnuto
	if (my_id % 2 == 0) {
		MPI_Send(&my_element, 1, MPI_DOUBLE, my_left, 0, MPI_COMM_WORLD);
		MPI_Send(&my_element, 1, MPI_DOUBLE, my_right, 0, MPI_COMM_WORLD);
		MPI_Recv(&left_val, 1, MPI_DOUBLE, my_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&right_val, 1, MPI_DOUBLE, my_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else {
		MPI_Recv(&left_val, 1, MPI_DOUBLE, my_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&right_val, 1, MPI_DOUBLE, my_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&my_element, 1, MPI_DOUBLE, my_right, 0, MPI_COMM_WORLD);
		MPI_Send(&my_element, 1, MPI_DOUBLE, my_left, 0, MPI_COMM_WORLD);
	}
	
	my_element = (my_element + left_val + right_val) / 3.0;	
	printf("Proces %d, vrijednost %f", my_id, my_element);
	
	MPI_Finalize();

	return 0;
}