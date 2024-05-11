#!/bin/bash


echo "#!/bin/bash" > ScLETD.slurm
echo >> ScLETD.slurm
echo "#SBATCH -J Omega_ScLETD" >> ScLETD.slurm
echo "#SBATCH -N "$1 >> ScLETD.slurm
echo "#SBATCH -n "`expr $1 \* 4` >> ScLETD.slurm
echo "#SBATCH -p normal" >> ScLETD.slurm
echo "#SBATCH -c 4" >> ScLETD.slurm
echo "#SBATCH --gres=dcu:4" >> ScLETD.slurm
echo "#SBATCH -o log/%J_$1.out" >> ScLETD.slurm
echo "#SBATCH -e log/%J_$1.err" >> ScLETD.slurm
echo "mpirun -np "`expr $1 \* 4`" ./bin/ScLETD "$1 $2 >> ScLETD.slurm

  sbatch ScLETD.slurm


