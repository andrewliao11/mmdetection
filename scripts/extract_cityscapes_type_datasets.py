'''
Train on sim10k, and evaluate on cityscapes
python eval_cityscapes.py --shift --num_gpus 4
'''
import os
import datetime
import argparse

from pathlib import Path
from train_mix_sim10k_cityscapes import get_sim10k_dataset_name


def execute(cmd, dry_run=False):
    print(cmd)
    if not dry_run:
        os.system(cmd)


def dump_new_config(test_dataset_name):
    base_config_path = "configs/my_cfg/base_extract_feat_detr_r50_8x2_150e_sim10k_cityscapes_car.py"
    new_config_path = f"configs/my_cfg/_extract_feat_detr_r50_8x2_150e_{test_dataset_name}.py"


    with open(base_config_path) as f:
        cont = f.read()
        lines = cont.split("\n")
        for i, l in enumerate(lines):
            if l.startswith("    test="):
                if test_dataset_name.startswith("cityscapes"):
                    new_l = f"""    test=cityscapes_test_dataset"""
                elif test_dataset_name.startswith("sim200k"):
                    new_l = f"""    test=sim200k_dataset"""
                elif test_dataset_name.startswith("sim10k"):
                    new_l = f"""    test=sim10k_dataset"""
                lines[i] = new_l
            

            if "sim10k" in test_dataset_name:
                if l.startswith("""sim10k_data_root = os.environ["HOME"] + '/datasets/sim10k/'"""):
                    new_l = f"""sim10k_data_root = os.environ["HOME"] + '/datasets/{test_dataset_name}/'"""

                
            if "sim200k" in test_dataset_name:
                if l.startswith("""sim200k_data_root = os.environ["HOME"] + '/datasets/sim200k/'"""):
                    new_l = f"""sim200k_data_root = os.environ["HOME"] + '/datasets/{test_dataset_name}/'"""


            if l.startswith("        type='DETRHead',"):
                new_l = f"""        type='ExtractFeatDETRHead',"""
                lines[i] = new_l

            if l.startswith("    type='DETR',"):
                new_l = f"""    type='ExtractFeatDETR',"""
                lines[i] = new_l
                
                
    cont = "\n".join(lines)
    with open(new_config_path, "w") as f:
        f.write(cont)
    
    return new_config_path


def main():

    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--test_dataset", type=str, choices=["cityscapes", "sim200k", "sim10k"], required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--out", type=str)
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    parser.add_argument("--drop", type=str, default="no", help="[param]-[criterion(small,truncated,occluded)]")
    #parser.add_argument('--run-local', action='store_true', help="whether to run in local machine")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument("--samples_per_gpu", type=int, default=2)
    #parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()
    os.chdir("../")


    if args.test_dataset == "cityscapes":
        print("Preparing Cityscapes")
        print("Extract car only")
        test_dataset_name = "cityscapes_car"

        out_dir = Path(os.environ["HOME"])
        out_dir = out_dir / "datasets" / test_dataset_name
        os.makedirs(out_dir, exist_ok=True)

        cmd = f"python tools/dataset_converters/cityscapes.py /datasets/cityscapes --nproc 8 --out-dir {out_dir / 'annotations'} --car-only"
        execute(cmd, dry_run=args.dry_run)
        cmd = f"ln -s /datasets/cityscapes/leftImg8bit {out_dir}"
        execute(cmd, dry_run=args.dry_run)
    elif args.test_dataset == "sim200k":
        
        print("Prepare Sim200k")
        out_dir = Path(os.environ["HOME"])
        
        test_dataset_name = get_sim10k_dataset_name(args.shift, args.scale, args.drop)
        test_dataset_name = test_dataset_name.replace("sim10k", "sim200k")

        out_dir = out_dir / "datasets" / test_dataset_name
        os.makedirs(out_dir, exist_ok=True)

        cmd = f"python tools/dataset_converters/sim10k.py /datasets/sim200k --shift {args.shift} --scale {args.scale} --nproc 8 --out-dir {out_dir / 'annotations'} --subsample 0.01"
        execute(cmd, dry_run=args.dry_run)
        cmd = f"ln -s /datasets/sim200k/VOC2012/JPEGImages {out_dir}"
        execute(cmd, dry_run=args.dry_run)
    elif args.test_dataset == "sim10k":
        
        print("Prepare Sim10k")
        out_dir = Path(os.environ["HOME"])
        
        test_dataset_name = get_sim10k_dataset_name(args.shift, args.scale, args.drop)
        #test_dataset_name = test_dataset_name.replace("sim10k", "sim200k")

        out_dir = out_dir / "datasets" / test_dataset_name
        os.makedirs(out_dir, exist_ok=True)

        cmd = f"python tools/dataset_converters/sim10k.py /datasets/sim10k --shift {args.shift} --scale {args.scale} --nproc 8 --out-dir {out_dir / 'annotations'}  --subsample 0.2"
        execute(cmd, dry_run=args.dry_run)
        cmd = f"ln -s /datasets/sim10k/VOC2012/JPEGImages {out_dir}"
        execute(cmd, dry_run=args.dry_run)
        
    else:
        
        raise ValueError()
    

    print("Testing")
    config = dump_new_config(test_dataset_name=test_dataset_name)
    
    
    cmd = f"python tools/test.py {config} {args.checkpoint_path} --cfg-options data.samples_per_gpu={args.samples_per_gpu} --show"
    if args.out:
        cmd += f" --out {args.out}"
    execute(cmd, dry_run=args.dry_run)


if __name__ == '__main__':
    main()


