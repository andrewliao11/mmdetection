'''
Train on sim10k + cityscapes, and evaluate on cityscapes
python train_mix_sim10k_cityscapes.py --shift 0.2-left --scale 0.2-up --drop 1000-small --subsample 0.5 --num_gpus 4
'''
import os
import datetime
import argparse

from pathlib import Path


def execute(cmd, dry_run=False):
    print(cmd)
    if not dry_run:
        os.system(cmd)


def dump_new_config(src_dataset_name, tgt_dataset_name):
    base_config_path = "configs/my_cfg/base_finetune_detr_r50_8x2_150e_mix_sim10k_cityscapes_car.py"
    new_config_path = f"configs/my_cfg/_detr_r50_8x2_150e_mix_{src_dataset_name}_{tgt_dataset_name}.py"

    with open(base_config_path) as f:
        cont = f.read()
        lines = cont.split("\n")
        for i, l in enumerate(lines):
            if l.startswith("sim10k_data_root = "):
                new_l = f"""sim10k_data_root = os.environ["HOME"] + '/datasets/{src_dataset_name}/'"""
                lines[i] = new_l

            if l.startswith("cityscapes_data_root = "):
                new_l = f"""cityscapes_data_root = os.environ["HOME"] + '/datasets/{tgt_dataset_name}/'"""
                lines[i] = new_l
                
    cont = "\n".join(lines)
    with open(new_config_path, "w") as f:
        f.write(cont)
    
    return new_config_path


def get_sim10k_dataset_name(shift, scale, drop):

    dataset_name = "sim10k"
    if shift != "no":
        ratio, direction = shift.split("-")
        ratio = int(float(ratio)*100)
        dataset_name += f"_shift-{ratio}-{direction}"

    if scale != "no":
        ratio, direction = scale.split("-")
        ratio = int(float(ratio)*100)
        dataset_name += f"_scale-{ratio}-{direction}"

    if drop != "no":
        param, criterion = drop.split("-")
        if criterion == "small":
            param = int(param)
        elif criterion in ["truncated", "occluded"]:
            param = int(param*100)
        else:
            raise ValueError

        dataset_name += f"_drop-{param}-{criterion}"
    return dataset_name


def main():

    parser = argparse.ArgumentParser(description="Run training")
    # target dataset perturbation
    parser.add_argument("--subsample", default=1., type=float, help="ratio of sub-sampling target dataset")
    # source dataset perturbation
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    parser.add_argument("--drop", type=str, default="no", help="[param]-[criterion(small,truncated,occluded)]")
    parser.add_argument('--run-local', action='store_true', help="whether to run in local machine")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument("--samples_per_gpu", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()
    os.chdir("../")

    print("Preparing Cityscapes")
    print("Extract car only")
    
    out_dir = Path(os.environ["HOME"])

    tgt_dataset_name = "cityscapes_car"

    if args.subsample < 1.:
        tgt_dataset_name += f"_subsample-{int(args.subsample*100)}"

    out_dir = out_dir / "datasets" / tgt_dataset_name
    os.makedirs(out_dir, exist_ok=True)

    cmd = f"python tools/dataset_converters/cityscapes.py /datasets/cityscapes --nproc 8 --out-dir {out_dir / 'annotations'} --car-only --subsample {args.subsample}"
    #execute(cmd, dry_run=args.dry_run)
    cmd = f"ln -s /datasets/cityscapes/leftImg8bit {out_dir}"
    #execute(cmd, dry_run=args.dry_run)


    print("Prepare Sim10k")
    out_dir = Path(os.environ["HOME"])
    

    src_dataset_name = get_sim10k_dataset_name(args.shift, args.scale, args.drop)
    out_dir = out_dir / "datasets" / src_dataset_name
    os.makedirs(out_dir, exist_ok=True)

    cmd = f"python tools/dataset_converters/sim10k.py /datasets/sim10k --shift {args.shift} --scale {args.scale} --drop {args.drop} --nproc 8 --out-dir {out_dir / 'annotations'}"
    #execute(cmd, dry_run=args.dry_run)
    cmd = f"ln -s /datasets/sim10k/VOC2012/JPEGImages {out_dir}"
    #execute(cmd, dry_run=args.dry_run)

    

    print("Training")
    config = dump_new_config(src_dataset_name=src_dataset_name, tgt_dataset_name=tgt_dataset_name)
    
    
    effective_batch_size = args.num_gpus * args.samples_per_gpu
    base_lr = 1e-4      #  for 8 GPUs and 2 img/gpu
    lr = base_lr * effective_batch_size / 16

    
    ct = datetime.datetime.now()
    wandb_name = f"{ct.year}.{ct.month}.{ct.day}.{ct.hour}.{ct.minute}.{ct.second}"
    wandb_project = f"label-translation-detr-mix_{src_dataset_name}-{tgt_dataset_name}"
    
    if args.num_gpus == 1:
        cmd = f"python tools/train.py {config} --cfg-options sim10k_data_root={out_dir}/ data.samples_per_gpu={args.samples_per_gpu} optimizer.lr={lr}"
    else:
        cmd = f"bash ./tools/dist_train.sh {config} {args.num_gpus} --cfg-options sim10k_data_root={out_dir}/ data.samples_per_gpu={args.samples_per_gpu} optimizer.lr={lr}"

    if not args.run_local:
        cmd += " --work-dir /results"

    execute(cmd, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
