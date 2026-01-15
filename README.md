#  📈🚫🏆 There are no Champions in Supervised Long-Term Time Series Forecasting 

This repository contains the code accompanying our paper:

> **[There are no Champions in Supervised Long-Term Time Series Forecasting](https://openreview.net/forum?id=yO1JuBpTBB)**  
> **Transactions on Machine Learning Research (TMLR), January 2026**  
> Lorenzo Brigato*, Rafael Morand*, Knut Strømmen*, Maria Panagiotou, Markus Schmidt, Stavroula Mougiakakou  
> University of Bern, Switzerland 🇨🇭    
> \* Equal contribution


The repository provides the experimental framework, model implementations, datasets, and hyperparameter optimization workflows used to conduct the large-scale empirical study presented in the paper.

 📊 Our work systematically evaluates supervised long-term time series forecasting models across diverse datasets and experimental settings, highlighting the **lack of universally dominant architectures** and offering **recommendations for future research**, including:
 - Improving benchmarking practices
 - Reducing unsubstantiated claims
 - Increasing dataset diversity and revising guidelines for model selection
   
 For additional details, please refer to our manuscript.

---

## :book: Repository Overview

This codebase builds on top of the **Time Series Library (TSLib)** benchmark introduced at ICLR 2023:
- [[Time Series Library (TSLib)]](https://github.com/thuml/Time-Series-Library)
- [[ICLR 2023 paper]](https://arxiv.org/abs/2210.02186)

We extend TSLib to support the experimental design of our study, including:
- additional baseline models
- additional datasets
- reproducible hyperparameter optimization pipelines

This repository contains all modifications and additions required to reproduce the results reported in our paper.

### 🤖🗃️ Models and Datasets

Beyond the existing **DLinear**, **PatchTST**, **TimeMixer**, **iTransformer**, and **TimeXer** implementations, we added the following models:

- [x] **ModernTCN** - ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis [[ICLR 2024]](https://openreview.net/pdf?id=vpJMJerXHU) [[Code]](https://github.com/luodhhh/ModernTCN)
- [x] **S-Mamba** - Is Mamba Effective for Time Series Forecasting? [[Neurocomputing 2025]](https://arxiv.org/abs/2403.11144) [[Code]](https://github.com/wzhwzhwzh0921/S-D-Mamba)

- [x] **xLSTMTime** - xLSTMTime: Long-Term Time Series Forecasting with xLSTM [[MDPI 2024]](https://www.mdpi.com/2673-2688/5/3/71) [[Code]](https://github.com/muslehal/xLSTMTime)
- [x] **iPatch** - Our hybrid transformed-based model introduced in this paper as a proof-of-concept model

In addition to the already available datasets from TSLib, the repository supports:

- [x] **UTSD** (_subset_) - Timer: Generative Pre-trained Transformers Are Large Time Series Models [[ICML 2024]](https://arxiv.org/abs/2402.02368) [[Hugging Face]](https://huggingface.co/datasets/thuml/UTSD)
  - ``AustraliaRainfall``
  - ``BeijingPM25Quality``
  - ``KDD Cup 2018``
  - ``Pedestrian Counts``
  - ``TDBrain``
  - ``BenzeneConcentration``
  - ``MotorImagery``

### 🏗️ Experimental Infrastructure

This repository includes utilities for large-scale and reproducible experimentation:

- Automated hyperparameter optimization using Optuna
- Reproducible search spaces aligned with the paper
- Scripts for running large-scale sweeps and model-efficiency analyses

---

## :gear: Usage

### Prepare Data
You can obtain the well-preprocessed datasets **included in the original TSLib** from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library). In addition, you can obtain the well-preprocessed **UTSD datasets** for our extensions from [[Hugging Face]](https://huggingface.co/datasets/thuml/UTSD). Then place the downloaded data in the folder `./dataset`.
 
### Installation
1. Clone this repository.
   ```bash
   git clone https://github.com/AIHNlab/NoChamps
   cd NoChamps
   ```

2. Create a new Conda environment.
   ```bash
   conda create -n nochamps python=3.11
   conda activate nochamps
   ```

3. Install Core Dependencies.
   ```bash
   pip install -r requirements.txt
   ```

4. Install [[PyTorch]](https://pytorch.org/).

### Run HPO

Run hyperparameter optimization for long-term forecasting using the provided shell script.

```
   bash scripts/long_term_forecast/hp_search/script_no_champions_in_ltsf.sh \
   96 \
   run.py \
   hp_results.txt
   ```

To run different configurations, adapt the shell script and call it using the following pattern.
```
   bash scripts/long_term_forecast/hp_search/<script>.sh \
   <prediction_length> \
   <training_script> \
   <results_file>
   ```

### Run Efficiency Analysis

After completing HPO and generating the results file, you can evaluate efficiency metrics (e.g., FLOPs, training speed, etc.) for the same long-term forecasting models using the provided shell script.

```
   bash scripts/long_term_forecast/model_efficiency/script_no_champions_in_ltsf.sh \
   <results_file> \
   <task_name> \
   <path_to_dataset_config> \
   <batch_size> \
   <train_epochs>
   ```
Example arguments for the previous command, which corresponds to our default setup employed in the paper, are: ```hp_results.txt```, ```long_term_forecast```, ```./scripts/long_term_forecast/model_efficiency/dataset_configs.json```, ```1```, and ```1000```.

The script will generate a file containing the raw efficiency metrics per model and dataset by adding the pattern ```_efficiency``` to the <results_file>, e.g., ```hp_results_efficiency.txt```. Additionally, the script produces another log file that aggregates all the efficiency results across datasets and further computes the efficiency-weighted error metric ξ. For more details, please refer to ```./utils/exp_efficiency_analyser.py``` and our paper (Section 4.6).  

---

## :writing_hand: Citation

If you find this repository useful for your research, please consider citing our paper:

```
@article{
brigato2026there,
title={There are no Champions in Supervised Long-Term Time Series Forecasting},
author={Lorenzo Brigato and Rafael Morand and Knut Joar Str{\o}mmen and Maria Panagiotou and Markus Schmidt and Stavroula Mougiakakou},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=yO1JuBpTBB},
note={}
}
```

---

## 🙏 Acknowledgements

We thank the authors and contributors of TSLib for providing a high-quality and extensible benchmark that enabled this work.

