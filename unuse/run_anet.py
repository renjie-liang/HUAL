import os
import sys

max_pos_len = 100
gpu_idx = 0
suffix = "TTT"
for I in range(0, 6):
    newtrain_path = "./scripts_iter/data/anet_{}/train{}.json".format(suffix, I+1)
    cmd_renew = "python ./scripts_iter/iter_anet_{}.py {} {}".format(suffix, suffix, I)
    os.system(cmd_renew)
    print("-------------------------- UPDATE DONE -------------------------\n")


    dataset_dir = "./data/dataset/activitynet_{}{}/".format(suffix, I+1)
    test_path = "./data/dataset/activitynet_gt/test.json"
    train_path = "./data/dataset/activitynet_{}{}/train.json".format(suffix, I+1)
    data_pkl = "./datasets/activitynet_i3d_{}_{}{}.pkl".format(max_pos_len, suffix, I+1)
    test_result_path = "./results/activitynet/{}{}.pkl".format(suffix, I+1)

    os.system("mkdir {}".format(dataset_dir))
    os.system("cp {} {}".format(test_path, dataset_dir))
    os.system("cp {} {}".format(newtrain_path, train_path))
    os.system("rm {} {}".format(data_pkl, test_result_path))
    print("-------------------------- MOVE DONE -------------------------\n")


    cmd_run = "python main.py --task activitynet --max_pos_len {} --gpu_idx {} --suffix {}{}".format(max_pos_len, gpu_idx,suffix, I+1)
    print(cmd_run)     
    os.system(cmd_run)
    print("------------------ TRAIN DONE ----------------\n")

    cmd_test = "python main.py --task activitynet --max_pos_len {} --gpu_idx {} --mode eval_trainset --suffix {}{}".format(max_pos_len, gpu_idx, suffix, I+1)
    print(cmd_test)     
    os.system(cmd_test)
    print("------------------ TEST DONE ----------------\n")
    print("\n"*3)
print("ALL DONE!")