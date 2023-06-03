import os
import sys

gpu_idx = 0
config_file = "charades"
for I in range(0, 1):
    cmd_update = "python update_label/update_charades.py {}".format(I)
    print(cmd_update)
    os.system(cmd_update)
    print("-------------------------- UPDATE DONE -------------------------\n")

    cmd_train = "python main.py --task charades  --gpu_idx {} --suffix re{}".format(gpu_idx, I)
    print(cmd_train)     
    os.system(cmd_train)
    print("------------------ TRAIN DONE ----------------\n")

    cmd_test = cmd_train + " --mode test"
    print(cmd_test)     
    os.system(cmd_test)
    print("------------------ TEST DONE ----------------\n\n")
    print("ALL DONE!")
        
    
    
    
    
    
    
    
    
    newtrain_path = "./scripts_iter/data/charades_{}/train{}.json".format(suffix, I+1)
    cmd_update = "python ./scripts_iter/iter_charades_{}.py {} {}".format(suffix, suffix, I)
    print(cmd_update)
    os.system(cmd_update)
    print("-------------------------- UPDATE DONE -------------------------\n")
    dataset_dir = "./data/dataset/charades_{}{}/".format(suffix, I+1)
    test_path = "./data/dataset/charades_gt/test.json"
    train_path = "./data/dataset/charades_{}{}/train.json".format(suffix, I+1)
    data_pkl = "./datasets/charades_i3d_{}_{}{}.pkl".format(max_pos_len, suffix, I+1)
    test_result_path = "./results/charades/{}{}.pkl".format(suffix, I+1)

    os.system("mkdir {}".format(dataset_dir))
    os.system("cp {} {}".format(test_path, dataset_dir))
    os.system("cp {} {}".format(newtrain_path, train_path))
    os.system("rm {} {}".format(data_pkl, test_result_path))
    print("-------------------------- MOVE DONE -------------------------\n")



    cmd_run = "python main.py --task charades --max_pos_len {} --char_dim 50 --gpu_idx {} --suffix {}{}".format(max_pos_len, gpu_idx,suffix, I+1)
    print(cmd_run)     
    os.system(cmd_run)
    print("------------------ TRAIN DONE ----------------\n")

    cmd_test = "python  main.py --task charades --max_pos_len {} --char_dim 50 --gpu_idx {} --mode eval_trainset --suffix {}{}".format(max_pos_len, gpu_idx, suffix, I+1)
    print(cmd_test)     
    os.system(cmd_test)
    print("------------------ TEST TRAINSET DONE ----------------\n")
    print("\n"*3)
print("ALL DONE!")