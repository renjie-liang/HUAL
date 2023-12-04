import os
import re
import json
import numpy as np
import sys
from utils.utils_hual import miou_two_dataset

# suffix = "TTT"
suffix = sys.argv[1]
GT_PATH = './data/dataset/activitynet_gt/train.json'

for I in range(1, 7):
    train_path = "./data/dataset/activitynet_{}{}/train.json".format(suffix, I)
    iou = miou_two_dataset(GT_PATH, train_path)

    path = "./ckpt/activitynet/model_i3d_100_{}{}/model".format(suffix, I)

    best_path = os.path.join(path, "checkpoint")
    with open(best_path, "r") as f:
        a = f.readline()
        a = a.split(".")[0].split("_")[-1]
        best_epoch = a
        # print(best_epoch)
    eval_res = os.listdir(path)
    eval_res = [i for i in eval_res if i[-3:]=="txt"]
    eval_res = sorted(eval_res)

    eval_path = os.path.join(path, eval_res[-1])
    with open(eval_path, "r") as f:
        a = f.readlines()

    for i in range(len(a)):
        # print(i)
        if a[i].endswith("Step {}:\n".format(best_epoch)):
            best_line = a[i+1]
            R357 = re.findall(r"\s\d+\.?\d*", best_line)
            res_print = "{:.4f}\t{}".format(iou, "\t ".join(R357))
            print(res_print)