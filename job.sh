#!/bin/bash 

 

#SBATCH --nodes=1  

#SBATCH --mem=20G 

#SBATCH --job-name=RL_tun

#SBATCH --time=25:00:00 

#SBATCH --output=tun.txt 

#SBATCH --error=tun.err

#SBATCH --cpus-per-task=20 

#SBATCH --ntasks=1 

 

python algppo.py

python algppo1.py

python algppo2.py

python algppo3.py

python algppo4.py

