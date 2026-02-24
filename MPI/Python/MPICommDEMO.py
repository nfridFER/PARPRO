# Jednostavni primjer komunikacije

from mpi4py import MPI

comm = MPI.COMM_WORLD
# ukupan broj procesa
world_size =  comm.Get_size()
# moj redni broj
world_rank = comm.Get_rank()


if world_rank == 0:
    number = 0
    comm.send(number, dest=1)
    print(f"Proces {world_rank} salje broj {number} procesu {world_rank + 1}")
    number = comm.recv(source=world_size - 1)
    print(f"Proces {world_rank} prima broj {number} od procesa {world_size - 1}")
else:
    number = comm.recv(source=world_rank - 1)
    print(f"Proces {world_rank} prima broj {number} od procesa {world_rank - 1}")
    number += 1
    comm.send(number, dest=(world_rank + 1) % world_size)
    print(f"Proces {world_rank} salje broj {number} procesu {(world_rank + 1) % world_size}")
