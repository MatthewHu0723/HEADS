# HEADS

The repository of DASFAA 2026 Article "Smoothing Sparsity Drift in Irregular Multivariate Time Series Forecasting Needs Temporal Observation Density"
We implement HEADS on the environment of Pytorch=3.9 and CUDA=12.1.

## Catalog
```
.
└─HEADS
  ├─HEADS
    ├─models                            Storing HEADS model and baselines
    ├─scripts                           Shell scripts
    ├─run.py                            Core training file
    ├─run_density_prep.py               Core preprocessing file
  ├─data                                Storing raw/preprocessed dataset files 
  ├─lib                                 Dataset-related functions and utils
  ├─README.md
  ├─requirements.txt                    Required Environment packages 
```

## How to Run

### 1. Prepare environment

```
pip install -r requirements.txt
```

### 2. Download datasets

For dataset *PhysioNet2012*, *Human Activity* and *USHCN*, there is no need for extra download.

For dataset *MIMIC-III*, because of the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciii/view-dua/1.4/), you must be credentialed first to download the offical dataset from [here](https://physionet.org/content/mimiciii/1.4/). After that, you can either download the raw dataset and follow the preprocessing pipline in [Neural Flows](https://github.com/mbilos/neural-flows-experiments) (like our choice), or directly download the preprocessed dataset at [MIMIC-III-Ext-tPatchGNN](https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/).

### 3. Preprocessing

We separate the preprocessing and training process to avoid processing dataset for every training time. Run  

```
bash scripts/run_prep_all.sh
```

to preprocess all the dataset.

### 4. Training

Run  

```
bash scripts/run_all.sh
```

to train all the three baseline on four dataset(with 5 seeds on default setting). 

You can adapt the script file and _run.py_ to train on a specific setting.

