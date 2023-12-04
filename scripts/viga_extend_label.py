import json
import numpy as np
import sys
sys.path.append("..")
from utils.runner_utils import calculate_iou

with open('../data/anet_viga/train_old.json', mode='rb') as f:
    data = json.load(f)
    
new_data = []
factor = 0.4
IOU = []
for vid in data:
    records = data[vid]
    duration = records["duration"]
    for time_gt, sentence, glance in zip(records["timestamps"], records["sentences"], records["glance"]):
        new_stime = max(glance - duration * factor / 2, 0)
        new_etime = min(glance + duration * factor / 2, duration)

        new_data.append([vid, duration, [new_stime, new_etime], sentence])
        IOU.append(calculate_iou([new_stime, new_etime], time_gt))
print(factor)
print(len(IOU), np.mean(IOU))

with open('../data/anet_viga/train.json', mode='w') as f:
    json.dump(new_data, f)