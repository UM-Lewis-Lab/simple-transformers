# Simple Transformers
This repository contains simplified example code to train and sample from transformer 
language models potentially across multiple GPUs using 
[🤗Transformers](https://github.com/huggingface/transformers) 
and [🤗Accelerate](https://github.com/huggingface/accelerate/).
## Prerequisites
We recommend you run this code in Docker using the provided `Dockerfile`.
In order to do this you must first [install Docker](https://docs.docker.com/get-docker/)
on the machine you will use to run your experiments.

## Installation
To get started, clone this repository and enter the cloned directory:
```bash
git clone https://github.com/UM-Lewis-Lab/simple-transformers.git
cd simple-transformers
```

Next, edit `.env` to change where cached files and model checkpoints will be stored.
If you are not sure, stick with the default values (but make sure you have enough disk
space to accomodate your checkpoints).

Finally, run the following command:
```bash
./run.sh all accelerate configure
```
Follow the prompts and answer the questions about your hardware (if you are not sure,
stick with the default answers).

# Usage
All of the scripts in this repository can be run via Docker using the `run.sh` script,
which can be used as follows:

```bash
./run.sh <gpus> <command> <script> <arguments>
```
- `<gpus>` is either `all` to use all GPUs or a comma separated list of device IDs
(e.g. `0,1,3` to use the first, second, and fourth GPU on the machine). You can view
a list of the machine's GPUs by running `nvidia-smi`.
- `<command>` should be either `accelerate launch` for training or `python` for everything else
(see examples below).
- `<script>` is the script you want to run (e.g. `train.py`).
- `<arguments>` will be passed to the `<script>` (see the `ArgumentParser` defined near the top
of each script for a list of accepted arguments and their functions).

⚠️**NOTE** The scripts only have access to files that are either inside the cloned repository
or inside one of the directories defined in `.env`. Either place your files inside the cloned
repository, or edit `run.sh` to mount the folder that contains them:
```bash
...
--mount "src=<your-source>,target=<your-target>,type=bind" \
...

```
where `<your-source>` is the path on machine with Docker installed, and `<your-target>` is the
path you want to mount it to inside the container.

## 📚Preparing a dataset
This example assumes you have a text file in which each line is a separate unit of text
(e.g. a sentence or paragraph) that the model should be trained on.
Before we can use the data to train a model, it must be pre-processed (tokenized, split
into managable chunks for the model, and split into train/test sets). To do this, run:

```bash
./run.sh all python prepare_datasets.py <path/to/your/file.txt> <path/to/write/processed/data.arrow>
```

We will use the resuling `.arrow` file to train our model.

## 🏋️Training a model
Once you have prepared the data, you can train a model using:
```bash
./run.sh all accelerate launch train.py <run_name> <path/to/data.arrow>
```
Where `run_name` is a string that identifies the current run (this will be used to e.g. name the 
directory where checkpoints will be stored).
See `train.py` for a list of additional options and
their functions.


## 💬Sampling from a trained model
After training a model, we can load on of the saved checkpoints and sample from it. First, identify
which checkpoint you would like to sample from. To get the name of the latest checkpoint you can run
```bash
./run.sh all /bin/bash -c 'ls /checkpoints/<run_name> | grep _ | sort -r | head -n 1'
```

Once you know the name of your checkpoint (e.g. `114_12`), you can sample from the model:

```bash
./run.sh all python sample.py --checkpoint /checkpoints/<run_name>/<checkpoint_name>
```

By default, the script will perform unconditional sampling, but you can provide a 
prompt as an argument: `--prompt 'This is an example prompt'`,
or by providing a CSV file with a list of prompts `--prompt_file path/to/file.csv`.
See `sampling.py` for a list of additional options and their functions.
