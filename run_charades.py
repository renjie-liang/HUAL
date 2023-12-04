import os
from utils.utils_hual import generate_configs
from utils.data_gen import gen_train_data_cache_path
# renew label -> train model -> test model -> ...
gpu_idx = 0
task = "charades"
base_configs_path = "./configs/charades/SeqPAN.yaml"
for I in range(1, 2):
    SUFFIX = "RE{}".format(I)

        # ----------------- RENEW LABEL -----------------
    renew_cmd = "python update_label/update_charades.py {} {}".format(task, I)
    print(renew_cmd)
    os.system(renew_cmd)
    print("----------------- RENEW LABEL -----------------\n\n")


    new_config_path, configs = generate_configs(base_configs_path, task, I)
    train_cmd = f"python main.py --config {new_config_path} --gpu_idx {gpu_idx}  --suffix {SUFFIX}"
    train_data_cache_path = gen_train_data_cache_path(configs)
    print(train_data_cache_path)
    # os.system("rm ./data_pkl/{}_i3d_64_{}.pkl".format(task, SUFFIX))
    
    print(train_cmd)
    # os.system(train_cmd)
    # print("----------------- TRAIN MODEL ----------------- \n\n")

    # test_cmd = train_cmd + " --mode test_save"
    # print(test_cmd)
    # os.system(test_cmd)
    # print("----------------- TEST MODEL -----------------\n\n")

