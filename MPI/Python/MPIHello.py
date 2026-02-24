#Hello World primjer

from mpi4py import MPI

comm = MPI.COMM_WORLD

#ukupan broj procesa
world_size =  comm.Get_size()

#moj redni broj
rank = comm.Get_rank()

print ("Hello world from process", rank, "out of", world_size, "processors.")


