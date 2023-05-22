import json
import numpy as np
import math
from utils_hual import time_to_index, index_to_time
import pickle
from utils_hual import miou_two_dataset, miou_two_dataset_idx, plt_hist, calculate_iou, infer_idx
from utils_hual import get_uncert_model, sigmoid, append_AP, center_width_gauss, fill_isactivate, get_segment
from utils_hual import get_distance_score, get_distance_score_shift
import sys
import random
import torch
from easydict import EasyDict
GT_PATH = './data/dataset/activitynet_gt/train.json'

F_renew = {"pos":{          #1.0, 2.0,
                "old":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "model":    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "distance": [2.0, 1.8, 1.6, 1.5, 1.5, 1.5], 
                },
            "neg":{
                "old":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "model":    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "distance": [2.0, 1.8, 1.6, 1.5, 1.5, 1.5], 
                },
            "uncert":[0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
            }
F_renew = EasyDict(F_renew)


def mask_activepoints(start_prob, end_prob, pos_idx, neg_idx, vlen):
    if len(pos_idx) == 0:
        for i in neg_idx:
            soft_mask = center_width_gauss(i, 0.3*vlen, vlen=vlen, max_vlen=len(start_prob))
            soft_mask = 1 - soft_mask
            start_prob = soft_mask * start_prob
            end_prob = soft_mask * end_prob
    else:
        # mask start prob
        lpos = min(pos_idx) 
        start_prob[lpos+1:] = 0 # +1 coz remain the point 
        tmp = [i for i in neg_idx if i < lpos]
        if len(tmp) > 0:
            start_prob[: max(tmp)+1] = 0

        # mask end prob
        rpos = max(pos_idx) 
        end_prob[:rpos] = 0
        tmp = [i for i in neg_idx if i > rpos]
        if len(tmp) > 0:
            end_prob[min(tmp):] = 0
    return start_prob, end_prob

def renew_label(old_idx, ap, sprob, eprob, vlen, max_vlen, I):
    pos_idx = ap['pos_idx']
    neg_idx = ap['neg_idx']

    global F_renew
    old_sprop =  center_width_gauss(old_idx[0], 0.5*vlen, vlen=vlen, max_vlen=max_vlen)
    old_eprop =  center_width_gauss(old_idx[1], 0.5*vlen, vlen=vlen, max_vlen=max_vlen)

    if len(pos_idx) > 0:
        a3, a2, a1 = F_renew.pos.distance[I], F_renew.pos.model[I], F_renew.pos.old[I]
        start_dis_score, end_dis_score = get_distance_score_shift(pos_idx, neg_idx, vlen=vlen, max_vlen=max_vlen, shift=0.3)
        start_score =  old_sprop*a1 + sprob*a2 + start_dis_score *a3
        end_score = old_eprop*a1 + eprob*a2 + end_dis_score*a3
        
        start_score, end_score = mask_activepoints(start_score, end_score, pos_idx, neg_idx, vlen=vlen)
        sidx = np.argmax(start_score)
        eidx = np.argmax(end_score)

    else:
        a3, a2, a1 = F_renew.neg.distance[I], F_renew.neg.model[I], F_renew.neg.old[I]
        start_dis_score, end_dis_score = get_distance_score_shift(pos_idx, neg_idx, vlen=vlen, max_vlen=max_vlen, shift=0.9)
        start_score =  old_sprop*a1 + sprob*a2 + start_dis_score *a3
        end_score = old_eprop*a1 + eprob*a2 + end_dis_score*a3


        start_score, end_score = mask_activepoints(start_score, end_score, pos_idx, neg_idx, vlen=vlen)

        sprob = torch.from_numpy(start_score)
        eprob = torch.from_numpy(end_score)
        outer = torch.matmul(sprob.unsqueeze(1),eprob.unsqueeze(0))
        score_matrix = torch.zeros_like(outer)
        
        neg_idx = sorted(neg_idx+[-1, vlen])
        for i in range(len(neg_idx)-1):
            ll, rr = neg_idx[i], neg_idx[i+1]
            score_matrix[ll+1:rr, ll+1:rr] = outer[ll+1:rr, ll+1:rr]
        score_matrix = torch.triu(score_matrix, diagonal=0)
        _, sidx = torch.max(torch.max(score_matrix, dim=1)[0], dim=0)  # (batch_size, )
        _, eidx = torch.max(torch.max(score_matrix, dim=0)[0], dim=0)  # (batch_size, )
        sidx, eidx = sidx.item(), eidx.item()
    return [sidx, eidx]

def get_uncert_rank(data_old, data_GT, SeqPAN_prop):
    res = []
    for idx, sample in enumerate(data_old):
        vid, duration, _, _, old_ap = sample
        assert vid == SeqPAN_prop[idx]["vid"]
        assert vid == data_GT[idx][0]

        vlen =  SeqPAN_prop[idx]['v_len']

        pos_idx =old_ap['pos_idx']
        neg_idx =old_ap['neg_idx']
        sprob, eprob = SeqPAN_prop[idx]["prop_logits"]
        sprob, eprob = sigmoid(sprob), sigmoid(eprob)
        # sprob[vlen:], eprob[vlen:] = 0, 0 # ???
        max_vlen = len(sprob)

        gt_time = data_GT[idx][2]
        gt_idx = time_to_index(gt_time, duration, vlen)
        old_idx = time_to_index(data_old[idx][2], duration, vlen)

        uncert_model = get_uncert_model(SeqPAN_prop[idx]["prop_logits1"], SeqPAN_prop[idx]["prop_logits2"], vlen)
        uncert_dist = get_distance_score(pos_idx, neg_idx, vlen=vlen, max_vlen=max_vlen)
        uncert_frame = uncert_dist + uncert_model * F_renew.uncert[I]

        uncert_video = np.sum(uncert_model)
        record = {"idx": idx, 
                "gt_idx": gt_idx, 
                "old_idx": old_idx, 
                "old_ap": old_ap, 
                "vlen": vlen, 
                "max_vlen": max_vlen, 
                "duration": duration, 
                
                # "uncert_model": uncert_model, 
                # "uncert_dist": uncert_dist, 
                "uncert_frame": uncert_frame, 
                "uncert_video": uncert_video, 

                "sprob": sprob, 
                "eprob": eprob, 
                }

        res.append(record)
        res = sorted(res, key=lambda x: x["uncert_video"], reverse=False)
    return res





def main(old_path, new_path, prop_path, I):
    with open(prop_path, 'rb') as fp:
        SeqPAN_prop = pickle.load(fp)

    with open(old_path, mode='r') as f:
        data_old = json.load(f)

    with open(GT_PATH, mode='r') as f:
        data_GT = json.load(f)

    if len(data_old[0]) == 4:
        for i in range(len(data_old)):
            data_old[i].append({'pos_idx':[], 'neg_idx': []})

    IOU = [[], []]
    uncert_rank = get_uncert_rank(data_old, data_GT, SeqPAN_prop)

    
    for i in  range(math.ceil(len(uncert_rank) / 2)):
        record = uncert_rank[i]
        idx = record["idx"]
        gt_idx = record["gt_idx"]
        old_ap = record["old_ap"]
        duration = record["duration"]
        uncert_frame = record["uncert_frame"]
        sprob, eprob = record["sprob"], record["eprob"]
        vlen, max_vlen = record["vlen"], record["max_vlen"]
        uncert_video = record["uncert_video"]
        old_idx = record["old_idx"]

        obsert_point = int(np.argmax(uncert_frame))
        new_ap = append_AP(obsert_point, old_ap, gt_idx)
        new_idx = renew_label(old_idx, new_ap, sprob, eprob, vlen, max_vlen, I)
        new_time = index_to_time(new_idx, duration, vlen)

        data_old[idx][2] = new_time
        data_old[idx][4] = new_ap
        if len(new_ap['pos_idx']) != 0:
            IOU[0].append(calculate_iou(new_idx, gt_idx))
        else:
            IOU[1].append(calculate_iou(new_idx, gt_idx))
    print("{:.4f} | {:.4f}".format(np.mean(IOU[0]), np.mean(IOU[1])))
    with open(new_path, mode='w') as f:
        json.dump(data_old, f)


if __name__ == "__main__":
    import os
    
    suffix = sys.argv[1]
    I = int(sys.argv[2])
    # print(os.cur)
    print("abs path is %s" %(os.path.abspath(sys.argv[0])))

    print("------------------")
    old_path = "./scripts_iter/data/anet_{}/train{}.json".format(suffix, I)
    new_path = "./scripts_iter/data/anet_{}/train{}.json".format(suffix, I+1)
    prop_path = "./results/activitynet/{}{}.pkl".format(suffix, I)
    print(prop_path)
    print(old_path)
    print(new_path)
    main(old_path, new_path, prop_path, I)

    old_miou = miou_two_dataset(GT_PATH, old_path)
    new_miou = miou_two_dataset(GT_PATH, new_path)
    print("{:.4f} -> {:.4f}".format(old_miou, new_miou), end="\t")

    old_miou = miou_two_dataset_idx(GT_PATH, old_path, 64)
    new_miou = miou_two_dataset_idx(GT_PATH, new_path, 64)
    print("{:.4f} -> {:.4f}".format(old_miou, new_miou))

