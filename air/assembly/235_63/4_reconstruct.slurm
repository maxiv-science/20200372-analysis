#!/bin/sh

#SBATCH --partition=v100
#SBATCH --reservation=konstantin
#SBATCH --job-name=pynx
#SBATCH --time=00:20:00
#SBATCH --output=%j.log
pwd; hostname; date

rm raw/*.cxi

rm *.cxi
source /mxn/home/alebjo/analysis/20200372-analysis/source_me_for_pynx
pynx-id01cdi.py data=prepared.npz nb_run_keep=20 nb_run=1000 support_threshold=0.1,0.5
pynx-cdi-analysis.py *.cxi modes
mv *.cxi raw/
