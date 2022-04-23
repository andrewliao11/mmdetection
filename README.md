# Usage

```bash
python train_cityscapes.py --num_gpus 4
python train_cityscapes.py --car-only --num_gpus 4
# train on sim10k and evaluate on cityscapes_car
python train_sim10k_cityscapes.py --shift 0.2-left --scale 0.2-up --num_gpus 4
# train on sim10k + cityscapes_car and evaluate on cityscapes_car
python train_mix_sim10k_cityscapes.py --shift 0.2-left --scale 0.2-up --num_gpus 4
```

Steps to add another dataset:
1. scripts `train_{dataset}.py`
2. create base config at `configs/my_cfg/base_detr_r50_8x2_150e_{dataset}.py`
  - Change custom_hooks.0.wandb_init_kwargs.project=label-translation-detr-{dataset}
  - Change num_classes
  - Change data.train, data.val, data.test
