# Proficiency Estimation - Demonstrator Proficiency

# Pretrained Checkpoints

We provide ego and exo model checkpoints trained on EgoExo4D. 

| name | dataset | view | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | EgoExo4D | ego | 16 | 448 | 79.6 | 94.0 | [model](https://www.dropbox.com/s/6f0x172lpqy3oxt/TimeSformer_divST_16x16_448_K400.pyth?dl=0) |
| TimeSformer | EgoExo4D | exo | 16 | 448 | 79.6 | 94.0 | [model](https://www.dropbox.com/s/6f0x172lpqy3oxt/TimeSformer_divST_16x16_448_K400.pyth?dl=0) |

# Installation

Follow the installation instructions for TimeSformer from the original repo: https://github.com/facebookresearch/TimeSformer 

## Dataset Preparation

Download the EgoExo4D proficiency estimation takes and annotations into: 

```
../data
```

To generate the splits, run: 

```
data_processing/ego_exo/generate_demonstrator_proficiency_splits.ipynb
```

save the generated files to: TimeSformer/ego_exo_splits

## Training the ego model

Replace the PATH_TO_DATA_DIR and PATH_PREFIX in configs/EgoExo/TimeSformer_divST_16x16_448.yaml with the full path to TimeSformer/ego_exo_splits/448pFull and to the ego_exo data directory (../data), respectively.

Then, use the following command to train TimeSformer on the egocentric view:

```
python tools/run_net.py --cfg configs/EgoExo/TimeSformer_divST_16x16_448.yaml OUTPUT_DIR ./outputs/448pFull/ego DATA.CAMERA_VIEW ego
```

## Training the exo model


Use the following command to train TimeSformer on the egocentric view:

```
python tools/run_net.py --cfg configs/EgoExo/TimeSformer_divST_16x16_448.yaml OUTPUT_DIR ./outputs/448pFull/exo_all DATA.CAMERA_VIEW exo_all
```

## Evaluation

Use the following command for evaluation. Replace <VIEW> with the proper view ('ego', 'exo_all'), <PATH_PREFIX> with the relative path to the repo, and <BEST_CKPT> with the name of the checkpoint.
```

python tools/run_net.py --cfg configs/EgoExo/TimeSformer_divST_16x16_448.yaml OUTPUT_DIR ./outputs/448pFull/<VIEW> DATA.CAMERA_VIEW <VIEW> TEST.CHECKPOINT_FILE_PATH <PATH_PREFIX>/ProficiencyEstimation/TimeSformer/outputs/448pFull/<VIEW>/checkpoints/<BEST_CKPT>.pyth TRAIN.ENABLE False TEST.SAVE_RESULTS_PATH preds.pkl
```

## Finetuning

To finetune from the existing checkpoint, add the following line in the command line, or in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
TRAIN.FINETUNE True
```

# Environment

The code was developed using python 3.8.17. For training, we used one GPU compute node containing 8 Quadro RTX 6000 GPUs.

# Acknowledgements

This codebase is built on top of [TimeSformer](https://github.com/facebookresearch/TimeSformer) by [Gedas Bertasius](https://www.gedasbertasius.com/) and [Lorenzo Torresani](https://ltorresa.github.io/home.html). We thank the authors for releasing their code.
