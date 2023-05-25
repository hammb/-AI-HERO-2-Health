AI-HERO 2 Health Challenge on energy efficient AI - Instance Segmentation of growing cells in time-lapsed sequences
=============================================================================================

This repository provides the code for the baseline. It contains a full training pipeline in Pytorch Lighting including dataloading, training a U-Net and evaluating.
You are free to clone this baseline and build on it.
The repository also includes necessary bash scripts for submitting to the HOREKA Cluster. 

The following sections contain detailed explanations about the structure, how to adapt the templates and how to get going on HOREKA.


Table of contents
=================

<!--ts-->
   * [Data](#data)
   * [Structure of the Skeleton Code](#structure-of-the-skeleton-code)
   * [HOREKA Setup](#horeka-setup)
     * [Clone from Github](#clone-the-skeleton-code)
     * [Virtual Environment](#set-up-your-environment)
   * [Training on HOREKA](#training-on-horeka)
   * [Monitoring Jobs](#useful-commands-for-monitoring-your-jobs-on-horeka)
   * [Inference](#inference)
   * [Evaluation](#evaluation)
<!--te-->

# Data

The train data is available in the following workspace on the cluster:

    /hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/train

It consists of 3 time-lapsed sequences of growing cells taken with an inverted microscope. Each time sequence consists of 800 images. The spatial image resolution is 0.072 μm/px. 

<img src="readme_imgs/img.png" width="400" height="400"> <img src="readme_imgs/mask.png" width="400" height="400"> 


The three sequences (a, b, c) are saved in separate folders. Ground truth masks are stored in the corresponding ```_GT``` folders. The baseline code utilizes a and b for training and c for validation. However, you are free to split the data as you like. The directory structure is as follows:
```
.
├── a
│   ├── t0000.tif
│   ├── t0001.tif
│   ├── ...
│   ├── t0797.tif
│   ├── t0798.tif
│   └── t0799.tif
├── a_GT
│   ├── man_seg0000.tif
│   ├── man_seg0001.tif
│   ├── ...
│   ├── man_seg0797.tif
│   ├── man_seg0798.tif
│   └── man_seg0799.tif
├── b
│   ├── t0000.tif
│   ├── t0001.tif
│   ├── ...
│   ├── t0797.tif
│   ├── t0798.tif
│   └── t0799.tif
├── b_GT
│   ├── man_seg0000.tif
│   ├── man_seg0001.tif
│   ├── ...
│   ├── man_seg0797.tif
│   ├── man_seg0798.tif
│   └── man_seg0799.tif
├── c
│   ├── t0000.tif
│   ├── t0001.tif
│   ├── ...
│   ├── t0797.tif
│   ├── t0798.tif
│   └── t0799.tif
└── c_GT
    ├── man_seg0000.tif
    ├── man_seg0001.tif
    ├── ...
    ├── man_seg0797.tif
    ├── man_seg0798.tif
    └── man_seg0799.tif
```

# Structure of the skeleton code

The baseline implements a border - core segmentation approach. A U-Net is trained to learn a semantic segmentation of the regions that represent a core of an instance and regions that belong to the border of the individual instances. During postrocessing the semantic segmentation is then transformed to an instance segmentation.

The content of the different files is as follows:

- `dataset.py`: Implements a Pytorch Dataset that loads the challenge data, additionally preprocesses the data to obtain a border - core representation of the ground truth masks
- `unet.py`: Implements a U-Net in pytorch lightning, additionally implements the function ```predict_instance_segmentation_from_border_core``` that performs the postprocessing.
- `train.py`: Implements a training pipeline, logs metrics in a csv file, saves the current last checkpoint
- `inference.py`: Loads the model weights of a trained model and runs inference including the postprocessing to obtain an instance segmentation
- `eval.py`: Computes the Instance Dice for given directories containing predictions and ground truth data

Additionally, bash scripts for running the training, inference and evaluation are available.

# HOREKA Setup

The HOREKA cluster is organized in workspaces. Each group got its own workspace assigned that is named after your group ID (e.g. H1).
In this workspace you will develop your code, create your virtual environment, save models and preprocessed versions of data and so on.
Once you're logged in to HOREKA, your first step is going to your group workspace.
For the following steps please substitute `<YOUR_GROUP_NAME>` by your group ID.

    cd /hkfs/work/workspace/scratch/hgf_pdv3669-<YOUR_GROUP_NAME>

### Clone the skeleton code

Clone this repository to your workspace. 

    cd /hkfs/work/workspace/scratch/hgf_pdv3669-<YOUR_GROUP_NAME>
    git clone https://github.com/XXXXXX.git

### Set up your environment

Follow the instructions to create a virtual environment. Optionally, you can install the requirements.txt from this repo if you want to build on it. You can choose the python version by simply adapting ```python3.9``` to your desired version.

#### go to your workspace
    cd /hkfs/work/workspace/scratch/hgf_pdv3669-<YOUR_GROUP_NAME>

#### create virtual environment
    python3.9 -m venv health_baseline_env
    source health_baseline_env/bin/activate
    pip install -U pip
    pip install -r /hkfs/work/workspace/scratch/hgf_pdv3669-<YOUR_GROUP_NAME>/AI-HERO-Health/requirements.txt


# Training on HOREKA

Submitting to HOREKA is done via the `sbatch` command. It requires a bash script that will be executed on the nodes.
You can find the bash script that starts training the baseline model in this repository (`train.sh`). 
In the script you also see the defined sbatch flags. For GPU nodes use `--partition=accelerated`. If you only need a cpu for a certain job, you can submit to `--partition=cpuonly`.
You can adapt all other flags if you want. Find more information about `sbatch` here: https://slurm.schedmd.com/sbatch.html.

In the script you need to adapt the path to your group workspace in lines 11 and 16. Then submit your job via:

    sbatch train.sh


# Useful commands for monitoring your jobs on HOREKA

List your active jobs and check their status, time and nodes:

    squeue

A more extensive list of all your jobs in a specified time frame, including the consumed energy per job:

    sacct --format User,Account,JobID,JobName,ConsumedEnergy,NodeList,Elapsed,State -S 2023-05-2508:00:00 -E 2023-06-2116:00:00

Print the sum of your overall consumed energy (fill in your user ID):

    sacct -X -o ConsumedEnergy --starttime 2023-05-2508:00:00 --endtime 2023-06-2116:00:00 --user <YOUR USER ID> |awk '{sum+=$1} END {print sum}'

Open a new bash shell on the node your job is running on and use regular Linux commands for monitoring:

    srun --jobid <YOUR JOB ID> --overlap --pty /bin/bash
    htop
    watch -n 0.1 nvidia-smi
    exit  # return to the regular HOREKA environment

Cancel / kill a job:
    
    scancel <YOUR JOB ID>

Find more information here: https://wiki.bwhpc.de/e/BwForCluster_JUSTUS_2_Slurm_HOWTO#How_to_view_information_about_submitted_jobs.3F


# Inference

Once the border-core semantic segmentation model is trained the final instance segmentation is predicted using `inference.py`. The version in this repository uses the `val` split. For the final evaluation we will run your modified version on the test set to obtain your final predictions. Therefore, make sure to properly adapt the inference script to whatever instance segmentation method you are using so that it saves your predictions. Also, resize the predictions to 256x256. 

Adapt the paths to your group workspace and run it via:

    sbatch inference.sh

# Evaluation

To calculate the Instance Dice for your predictions you can use `eval.py`. In the baseline repository it evaluates on the sequence `c` so if you change the train/validation split and want to compute the score on other sequences you need to adapt the script accordingly. Note that for the final evaluation of the test set we will use our own version of this script.

Adapt the paths to your group workspace and run it via:

    sbatch eval.sh
