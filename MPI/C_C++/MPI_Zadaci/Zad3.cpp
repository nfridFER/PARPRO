#include <mpi.h>
#include <stdio.h>

int main(int argc, char * argv[]) {

	MPI_Init(NULL, NULL);

	int my_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int data;

	//proces 0 sluzi kao dirigent
	if (my_id == 0) {

		//cekaj da svi jave da su dosli do zadane tocke
		for (int i = 1; i < world_size; i++) {
			//redoslijed primanja nije vazan - ANY_SRC, samo je bitno da od svih procesa dode poruka
			MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		printf("-----------Primljene sve poruke.\n");
		fflush(NULL); // printaj odmah

		//javi svima da mogu nastaviti dalje
		for (int i = 1; i < world_size; i++) {
			MPI_Send(&my_id, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}
	}
	else {

		//radi nesto
		for (int n = 0; n < my_id; n++) {
			printf("Proces %d radi nesto...iteracija %d\n", my_id, n);
		}

		//kontrolna tocka - barrier
		printf("Proces %d dosao do kontrolne tocke.\n", my_id); fflush(NULL);
		MPI_Send(&my_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&data, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Proces %d nastavlja.\n", my_id); fflush(NULL);

	}


	MPI_Finalize();

	return 0;
}
