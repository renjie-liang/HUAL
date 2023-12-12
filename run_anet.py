import os
from utils.utils_hual import generate_configs
from utils.data_gen import gen_train_data_cache_path

gpu_idx = 0
task = "charades"
base_configs_path = "./configs/charades/SeqPAN.yaml"

for I in range(1, 2):
    SUFFIX = "re{}".format(I)

    # ----------------- RENEW LABEL -----------------
    renew_cmd = "python update_label/update_charades.py {} {}".format(task, I)
    print(renew_cmd)
    try:
        os.system(renew_cmd)
        print(f"Success in renewing labels: {task} {I}")
        
    except Exception as e:
        print(f"Error in renewing labels: {e}")

    try:
        new_config_path, configs = generate_configs(base_configs_path, task, I)
        configs.suffix = SUFFIX
        train_cmd = f"python main.py --config {new_config_path} --gpu_idx {gpu_idx}  --suffix {SUFFIX}"
        train_data_cache_path = gen_train_data_cache_path(configs)
        print(train_data_cache_path)
        os.system(f"rm {train_data_cache_path}")
        print(train_cmd)
        os.system(train_cmd)
        print(f"Success in training model: {task} {I}")
    except Exception as e:
        print(f"Error in training model: {e}")

    try:
        test_cmd = train_cmd + " --mode infer_trainset"
        print(test_cmd)
        os.system(test_cmd)
        print(f"Error in infering train set model: {task} {I}")
    except Exception as e:
        print(f"Error in infering train set model: {e}")
