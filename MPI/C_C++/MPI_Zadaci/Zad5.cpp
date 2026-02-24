#include <mpi.h>
#include <stdio.h>
#include <random>
#include <cmath>

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
	int x = distribution(gen);
	printf("Proces %d, broj %d\r\n", my_id, x); fflush(NULL);

	MPI_Barrier(MPI_COMM_WORLD); //samo radi ljepseg ispisa


	int temp_var;

	for (int i = 0; i < log2(world_size); i++) {
		printf("Proc %d, i: %d\n", my_id, i);	fflush(NULL);
		MPI_Barrier(MPI_COMM_WORLD); //samo radi ljepseg ispisa

		if (my_id%(int)pow(2, i+1) == 0) {
			int recv_id = my_id + pow(2, i);
			printf("Proc % d, recv_id: %d\n",my_id, recv_id); fflush(NULL);
			if (recv_id < world_size) {
				MPI_Recv(&temp_var, 1, MPI_INT, recv_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (temp_var < x) x = temp_var;
			}			
		}
		else if(my_id % (int)pow(2, i + 1) == pow(2, i)) {
			int send_id = my_id - pow(2, i);
			printf("Proc %d, send_id: %d\n", my_id, send_id); fflush(NULL);
			MPI_Send(&x, 1, MPI_INT, send_id, 0, MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD); //samo radi ljepseg ispisa
	}


	for (int i = log2(world_size); i > 0; i--) {
		printf("Proc %d, i: %d\n", my_id, i);	fflush(NULL);
		MPI_Barrier(MPI_COMM_WORLD); //samo radi ljepseg ispisa

		if (my_id % (int)pow(2, i) == 0) {
			int send_id = my_id + pow(2, i-1);
			printf("Proc % d, send_id: %d\n", my_id, send_id); fflush(NULL);
			MPI_Send(&x, 1, MPI_INT, send_id, 0, MPI_COMM_WORLD);
			
		}
		else if (my_id % (int)pow(2, i) == pow(2, i-1)) {
			int recv_id = my_id - pow(2, i-1);
			printf("Proc %d, recv_id: %d\n", my_id, recv_id); fflush(NULL);
			MPI_Recv(&x, 1, MPI_INT, recv_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		MPI_Barrier(MPI_COMM_WORLD); //samo radi ljepseg ispisa
		
	}

	printf("Proces %d, min: %d\n", my_id, x);


	return 0;
}
