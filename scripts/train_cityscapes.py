'''
Train on cityscapes, and evaluate on cityscapes
python train_cityscapes.py --subsample 0.5 --num_gpus 4
'''
import os
import datetime
import argparse

from pathlib import Path


def execute(cmd, dry_run=False):
    print(cmd)
    if not dry_run:
        os.system(cmd)


def dump_new_config(dataset_name):
    base_config_path = "configs/my_cfg/base_detr_r50_8x2_150e_cityscapes.py"
    new_config_path = f"configs/my_cfg/_detr_r50_8x2_150e_{dataset_name}.py"

    with open(base_config_path) as f:
        cont = f.read()
        lines = cont.split("\n")
        for i, l in enumerate(lines):
            if l.startswith("data_root = "):
                new_l = f"""data_root = os.environ["HOME"] + '/datasets/{dataset_name}/'"""
                lines[i] = new_l
                
    cont = "\n".join(lines)
    with open(new_config_path, "w") as f:
        f.write(cont)
    
    return new_config_path

    
def main():

    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--subsample", default=1., type=float, help="ratio of sub-sampling target dataset")
    parser.add_argument('--car-only', action='store_true', help="whether to extract car annotations only")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-local', action='store_true', help="whether to run in local machine")
    parser.add_argument("--samples_per_gpu", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    os.chdir("../")

    print("Preparing Cityscapes")
    
    out_dir = Path(os.environ["HOME"])
    dataset_name = "cityscapes"
    
    if args.car_only:
        print("Extract car only")
        dataset_name += "_car"

    
    if args.subsample < 1.:
        dataset_name += f"_subsample-{int(args.subsample*100)}"

    
    out_dir = out_dir / "datasets" / dataset_name
    cmd = f"python tools/dataset_converters/cityscapes.py /datasets/cityscapes --nproc 8 --out-dir {out_dir / 'annotations'} --subsample {args.subsample}"
    if args.car_only:
        cmd += " --car-only"
    execute(cmd, dry_run=args.dry_run)

    cmd = f"ln -s /datasets/cityscapes/leftImg8bit {out_dir}"
    execute(cmd, dry_run=args.dry_run)


    print("Training")
    config = dump_new_config(dataset_name=dataset_name)
        
    

    effective_batch_size = args.num_gpus * args.samples_per_gpu
    base_lr = 1e-4      #  for 8 GPUs and 2 img/gpu
    lr = base_lr * effective_batch_size / 16

    
    ct = datetime.datetime.now()
    wandb_name = f"{ct.year}.{ct.month}.{ct.day}.{ct.hour}.{ct.minute}.{ct.second}"
    
    if args.num_gpus == 1:
        cmd = f"python tools/train.py {config} --cfg-options data_root={out_dir}/ data.samples_per_gpu={args.samples_per_gpu} optimizer.lr={lr} custom_hooks.0.wandb_init_kwargs.name={wandb_name}"
    else:
        cmd = f"bash ./tools/dist_train.sh {config} {args.num_gpus} --cfg-options data_root={out_dir}/ data.samples_per_gpu={args.samples_per_gpu} optimizer.lr={lr} custom_hooks.0.wandb_init_kwargs.name={wandb_name}"

    if args.car_only:
        cmd += " custom_hooks.0.wandb_init_kwargs.project=label-translation-detr-cityscapes_car"

    if not args.run_local:
        cmd += " --work-dir /results"

    execute(cmd, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
