import os


ngc_id_info = {
    2838351: "S + T", 
    2838290: "shift-30-left", 
    2838292: "shift-60-left", 
    2838293: "shift-30-up", 
    2838294: "shift-60-up", 
    2838295: "scale-20-up", 
    2838296: "scale-20-down", 
}

for ngc_id, setting in ngc_id_info.items():
    print(setting)
    cmd = f"python train.py {ngc_id}"
    os.system(cmd)
    print("="*50)
