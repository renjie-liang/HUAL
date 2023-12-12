import os
from utils.utils_hual import generate_configs
from utils.data_gen import gen_train_data_cache_path

gpu_idx = 0
task = "charades"
base_configs_path = "./configs/charades/SeqPAN.yaml"

for I in range(1, 4):
    SUFFIX = "re{}".format(I)

    # ----------------- update label -----------------
    renew_cmd = "python update_label.py {} {}".format(task, I)
    print(renew_cmd)
    if os.system(renew_cmd) != 0:
        print(f"Error in renewing labels for {task} {I}")
        break
    print(f"Success in renewing labels: {task} {I}")


    # -----------------Train model -----------------
    new_config_path, configs = generate_configs(base_configs_path, task, I)
    configs.suffix = SUFFIX
    train_cmd = f"python main.py --config {new_config_path} --gpu_idx {gpu_idx}  --suffix {SUFFIX}"
    train_data_cache_path = gen_train_data_cache_path(configs)
    print(train_data_cache_path)
    os.system(f"rm {train_data_cache_path}")
    print(train_cmd)
    if os.system(train_cmd) != 0:
        print(f"Error in training model for {task} {I}")
        break
    print(f"Success in training model: {task} {I}")


    # -----------------Infering train set -----------------
    test_cmd = train_cmd + " --mode infer_trainset"
    print(test_cmd)
    if os.system(test_cmd) != 0:
        print(f"Error in infering train set model for {task} {I}")
        break
    print(f"Success in infering train set model: {task} {I}")
