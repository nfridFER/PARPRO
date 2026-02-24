#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <random>


int main(int argc, char * argv[]) {

	MPI_Init(NULL, NULL);

	int my_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);


	//svaki proces ima neki random podatak manji od 100
	std::random_device rd;
	std::mt19937 gen(rd() + my_id); 
	std::uniform_int_distribution<int> distribution(1, 100);
	int my_element = distribution(gen);
	printf("Proces %d, broj %d\r\n", my_id, my_element);
	
	int my_left = (my_id + world_size - 1) % world_size;
	int my_right = (my_id + 1) % world_size;

	

	if (my_id == 0) { //prvi
		MPI_Send(&my_element, 1, MPI_INT, my_right, 0, MPI_COMM_WORLD);
		MPI_Recv(&my_element, 1, MPI_INT, my_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("MAX: %d", my_element);
	}
	else if (my_id>0 && my_id < (world_size - 1)) { //srednji
		int temp_val;
		MPI_Recv(&temp_val, 1, MPI_INT, my_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (temp_val > my_element)
			my_element = temp_val;
		MPI_Send(&my_element, 1, MPI_INT, my_right, 0, MPI_COMM_WORLD);
		MPI_Recv(&my_element, 1, MPI_INT, my_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&my_element, 1, MPI_INT, my_left, 0, MPI_COMM_WORLD);
	}
	else { //zadnji
		int temp_val;
		MPI_Recv(&temp_val, 1, MPI_INT, my_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (temp_val > my_element)
			my_element = temp_val;
		MPI_Send(&my_element, 1, MPI_INT, my_left, 0, MPI_COMM_WORLD);
	}
	

	MPI_Finalize();

	return 0;
}

