# PySSL_SERS

Implementation of PySSL_SERS

# Requirements and installation

We will set up the environment using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html). Clone the current repo

    git clone https://github.com/junho1ee/pyssl_sers.git

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, cuda versions or cpu only versions):

    conda create --name pyssl_sers python=3.9
    conda activate pyssl_sers
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install lightning==2.0.9

# Dataset

## Download
If you want to train our models with the data then:
1. download it from links:
    * [Bacteria-ID](https://github.com/csho33/bacteria-ID/)
    * [COVID-19](https://springernature.figshare.com/articles/dataset/Data_and_code_on_serum_Raman_spectroscopy_as_an_efficient_primary_screening_of_coronavirus_disease_in_2019_COVID-19_/12159924)
2. unzip the directory and place it into `data` such that you have the path `data/bacteria-id/org` and `data/covid/org`

## Preprocess
If you want to preprocess the data then:

    python data_preprocess_bacteria.py
    python data_preprocess_covid.py

# Running the code
## Pretraining a new model

    python lightning_pretrain_ssl.py --pre byol --augtype phys

## Finetuning and test a pretrained model

    python lightning_finetune_pred.py --pre byol --task class30 --augtype phys --fold 0


# Code references

Our implementation referred to thre following repositories:

- PySSL (https://github.com/giakou4/pyssl)
- fairseq-signals (https://github.com/Jwoo5/fairseq-signals)

# Contact
If you have any questions, please contact
- awer2072@gmail.com
