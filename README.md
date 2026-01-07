# IndoorUAV-Agent

**IndoorUAV: Benchmarking Vision-Language UAV Navigation in Continuous Indoor Environments.** This project contains the finetuning and evaluation code of our AAAI 2026 paper.

[[Paper](https://arxiv.org/abs/2512.19024)] [[Dataset](https://www.modelscope.cn/datasets/valyentine/Indoor_UAV)]

## Release
- [x] Pre-trained IndoorUAV-Agent model
- [x] Online evaluation scripts for VLA and VLN tasks
- [x] Metric evaluation scripts
- [x] Fine-tuning configuration and guidelines

## Contents
- [Model Download](#model-download)
- [Online Evaluation](#online-evaluation)
  - [Environment Setup](#environment-setup)
  - [Running Evaluation](#running-evaluation)
- [Metric Evaluation](#metric-evaluation)
- [Fine-Tuning](#fine-tuning)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Model Download
Download the pre-trained IndoorUAV-Agent model from ModelScope:

| Model | Download |
|-------|----------|
| IndoorUAV-Agent | [checkpoint](https://modelscope.cn/models/valyentine/IndoorUAV-Agent) |

This model is obtained by fine-tuning the pi0 model for 30k steps using the VLA part (15k episodes) of the IndoorUAV dataset.

## Online Evaluation
Because the simulator environment and the inference model environment are not compatible, you need to set up two separate environments.

### Environment Setup
1. **Simulator Environment**  
   Follow the instructions in [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/main) to install the Habitat simulator.

2. **Inference Model Environment**  
   Set up the environment following [openpi](https://github.com/Physical-Intelligence/openpi).

### Running Evaluation
1. **Download the dataset** from [ModelScope](https://modelscope.cn/datasets/valyentine/Indoor_UAV).

2. **Modify configuration files**:
   - In [vla_controller.py](online_eval/vla_eval/vla_controller.py) and [vln_controller.py](online_eval/vla_eval/vln_controller.py), set `INDOOR_UAV_BASE` to your dataset path.
   - In [model_runner.py]((online_eval/vla_eval/model_runner.py)), update `checkpoint_dir` to the path of the downloaded model.
   - In [utils.py]((online_eval/vla_eval/utils.py)), set the scene path to point to your `scene_datasets` folder.

3. **Run evaluation scripts**:
   ```bash
   ./online_eval/vla_eval.sh   # for VLA task
   ./online_eval/vln_eval.sh   # for VLN task
   ```
      To save disk space by downloading part of the dataset, you can test on a subset of episodes by modifying:
   - `TEST_VLA_FILE` in [vla_controller.py](online_eval/vla_eval/vla_controller.py)
   - `TEST_VLN_FILE` in [vln_controller.py](online_eval/vla_eval/vln_controller.py)
   
   and the corresponding JSON files.

## Metric Evaluation
After batch testing, you can compute quantitative metrics using:

```bash
python eval_metric/vla_metric.py   # for VLA task
python eval_metric/vln_metric.py   # for VLN task
```
Before running, set the `trajectories_dir` parameter in each script to the actual path where trajectories are stored.

## Fine-Tuning
### Data Preparation
Convert your data to RLDS format using [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder).

### Fine-Tuning Steps
Refer to the section **"Fine-Tuning Base Models on Your Own Data"** in the [openpi](https://github.com/Physical-Intelligence/openpi) repository.

We provide our fine-tuning configuration files in the `config/` folder for reference.

## Citation
If you find this work useful, please cite:
```
@article{liu2025indooruav,
title={IndoorUAV: Benchmarking Vision-Language UAV Navigation in Continuous Indoor Environments},
author={Liu, Xu and Liu, Yu and Qiu, Hanshuo and Qirong, Yang and Lian, Zhouhui},
journal={arXiv preprint arXiv:2512.19024},
year={2025}
}
```

## Acknowledgments
Our code is built upon:
- [pi0 (openpi)](https://github.com/Physical-Intelligence/openpi)
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/main)

For any questions, please contact Yu Liu at [yuliu_@hust.edu.cn](mailto:yuliu_@hust.edu.cn).
