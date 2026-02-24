#include <mpi.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include <chrono>

int main(int argc, char * argv[]) {

	MPI_Init(NULL, NULL);
    
	int my_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	std::chrono::high_resolution_clock::time_point start_time, end_time;

	if (my_id == 0) {
		start_time = std::chrono::high_resolution_clock::now();
	}


	long b_tocaka = 10000000; // bilo koji veliki broj – sto veci to preciznije
	int b_tocaka_krug = 0;
	int p = world_size;
	int b_tocaka_p = b_tocaka / p;

	/*priprema za generiranje random brojeva od 0  do 10*/
	std::random_device rd;
	std::mt19937 gen(rd() + my_id);
	std::mt19937 gen2(rd() + 2*my_id);
	std::uniform_int_distribution<int> distribution(0, 10);

	for (int i = 0; i < b_tocaka_p; i++) {
		int x= distribution(gen);
		int y= distribution(gen2);

		if ((x * x + y * y) <= 100) {
			b_tocaka_krug++; //točka unutar kruga
		}
	}

	if (my_id == 0) { //ako voditelj
		//primi b_tocaka_krug od svih radnika
		int tmp = 0;
		for (int n = 1; n < p; n++) {
			MPI_Recv(&tmp, 1, MPI_INT, n, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			b_tocaka_krug += tmp;
		}

		double pi = 4.0 * (double)b_tocaka_krug / (double)b_tocaka;
		
		end_time = std::chrono::high_resolution_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		
		printf("PI: %f, vrijeme: %d\r\n", pi, time); fflush(NULL);

	}
	else {	//ako radnik posalji voditelju b_tocaka_krug
		MPI_Send(&b_tocaka_krug, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}


	MPI_Finalize();

	return 0;
}



