# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
# export PATH="/usr/local/nvidia/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
# source activate py35
# Change to the directory in which your code is present
# cd run.sh
# Run the code. The -u option is used here to use unbuffered
# cd storage/home/gsk1692/work/adg_project/ 
python -u storage/home/gsk1692/work/adg_project/dual_att_gru/main.py > /scratch/scratch6/gokul/adg_project_data/out.txt