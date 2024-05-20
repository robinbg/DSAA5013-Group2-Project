# DSAA5013-Group2-Project

## File explanation

- model.py: A wrapper to make EffcientNetV2 trainable by Trainer
- utils.py: Cutout augmentation
- train.py train

## How to run
python train.py --model-path MODEL_PATH --pretrained PRETRAINED

MODEL_PATH: path to pretrained weight(architecture)
USE timm/tf_efficientnetv2_s.in1k
    timm/tf_efficientnetv2_m.in1k
    timm/tf_efficientnetv2_l.in1k
to load different models

PRETRAINED: True or False
To control whether loading the pretrained weight into the model.
If False, the model will be initialized randomly.
