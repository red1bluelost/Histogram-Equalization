#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=nvprof_heq
#SBATCH --reservation=GPU-CLASS-FL20
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:1
#SBATCH --output=nvprof_heq.%j.out

module load opencv/3.4.3-contrib

cd /scratch/$USER/GPUClassF20/heq/

set -o xtrace
#nvprof --metrics gld_requested_throughput, gst_requested_throughput, gst_throughput, gld_throughput, gld_efficiency, gst_efficiency, stall_memory_dependency, gld_transactions_per_request, gst_transactions_per_request ./heq input/bridge.png
nvprof ./heq input/bridge.png

