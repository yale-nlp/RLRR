# RefEval

This repository is for our work "References Improve LLM Alignment in Non-Verifiable Domains."

## Outline

- [How to Run](#how-to-run)
    - [Installation and Requirements](#installation-and-requirements)
    - [Running Experiments](#running-experiments)
- [Model Checkpoints](#model-checkpoints)
    - [Best Checkpoints](#best-checkpoints)
    - [All Checkpoints](#all-checkpoints)
- [File Structure](#file-structure)
    - [Files](#files)
    - [Directories](#directories)

## How to Run

### Installation and Requirements
Please run `pip install -r requirements.txt` to install the required packages.

For training, you will need at least 8 GPUs with 48GB of memory each. The code is tested on a machine with 8 NVIDIA A6000 Ada GPUs.

### Running Experiments
To run the self-improvement training experiment, please use the following command: `bash self_improve.sh`.

The script [`self_improve.sh`](self_improve.sh) performs preference optimization using LLM-judge to self-improve with the following steps:
1. Sampling candidate outputs from the LLM.
2. Scoring the candidate outputs using the model itself as a judge.
3. Data processing and precomputing the log probabilities of the output pairs.
4. Training: traning the LLM using DPO.


## File Structure

### Files

- [`self_improve.sh`](self_improve.sh): Script for running the self-improvement training experiment.
- [`data_processing.py`](data_processing.py): Contains the code for post-processing the preference model annotations into training data.
- [`data_utils.py`](data_utils.py): Utility functions for training data loading.
- [`get_logprobs.py`](get_logprobs.py): Script for extracting log probabilities from an LLM/policy.
- [`losses.py`](losses.py): Loss functions.
- [`dpo.py`](dpo.py): DPO training.
- [`mle.py`](mle.py): MLE training.
- [`sampling.py`](sampling.py): Sampling candidate outputs from an LLM.
- [`scoring.py`](scoring.py): Scoring output pairs using a preference model.
- [`utils.py`](utils.py): Utility functions.
- [`vllm_model.py`](vllm_model.py): VLLM model definition.
- [`deepspeed.conf'](deepspeed_config.json): Deepspeed configuration file.

### Directories
- [`data`](data): Contains the training data, which will be provided in the future.
- [`exps`](exps): Contains the results of the experiments. A new directory is created for each experiment, with the name specified in `self_improve.sh`.
- [`prompts`](prompts): Contains the prompts used by the LLM-judge.
