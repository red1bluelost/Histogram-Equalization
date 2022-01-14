#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec_heq
#SBATCH --reservation=GPU-CLASS-FL20
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:1
#SBATCH --output=exec_heq.%j.out

module load opencv/3.4.3-contrib

cd /scratch/$USER/GPUClassF20/heq/

set -o xtrace
./heq input/bridge.png
./heq input/flower.jpg
./heq input/fseprd531122.jpg
./heq input/Geotagged_articles_wikimap_RENDER_ca_huge.png
./heq input/in-2.jpg
./heq input/in.jpg
./heq input/Wikidata_Map_April_2016_Huge.png

