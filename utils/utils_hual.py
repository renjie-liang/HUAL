from copyreg import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import math
import os
import shutil
from omegaconf import OmegaConf


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    if (union[1] - union[0]) == 0.0:
        return 0.0
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def miou_two_dataset(path1, path2):
    with open(path1, mode='r') as f:
        data1 = json.load(f)
    with open(path2, mode='r') as f:
        data2 = json.load(f)
    assert len(data1) == len(data2)

    miou = []
    for x1, x2 in zip(data1, data2):
        assert x1[0] == x2[0]
        iou = calculate_iou(x1[2], x2[2])
        miou.append(iou)
    return np.mean(miou)


def fill_isactivate(pos_idx, neg_idx, vlen, max_vlen):
    isactive = np.zeros(max_vlen)
    if len(pos_idx) > 0:
        # file positive area
        ll = min(pos_idx)
        rr = max(pos_idx)
        isactive[ll: rr+1] = 1

        # fill negative areas
        ll_negs = [i for i in neg_idx if i < ll]
        rr_negs = [i for i in neg_idx if i > rr]

        if len(ll_negs)>0:
            isactive[: max(ll_negs)+1] = -1
        if len(rr_negs)>0:
            isactive[ min(rr_negs):] = -1

    else:
        for i in neg_idx:
            isactive[i] = -1
    isactive[vlen:] = -100
    return isactive



# gene distance score
def get_segment(isactive):
    segment_list = []
    i = 0
    isactive_loop = isactive.tolist() + [-100]
    while i < len(isactive_loop):
        if  isactive_loop[i] == 0:
            for j in range(i+1, len(isactive_loop)):
                if isactive_loop[j] != 0.0 :
                    segment_list.append([i, j-1])
                    i = j+1
                    break
        else:
            i += 1
    return segment_list


def center_width_gauss(center, width, vlen, max_vlen):
    sigma = 0.4
    x = np.linspace(-1, 1, num=max_vlen,  dtype=np.float32)
    sig = vlen / max_vlen
    sig *= width / vlen * sigma
    u = (center / (max_vlen-1)) * 2 - 1
    weight = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
    weight /= np.max(weight)
    weight *= width/vlen #/ 2 + 0.5
    weight[vlen:] = 0.0
    return weight


def get_distance_score(pos_idx, neg_idx, vlen, max_vlen):
    isactive = fill_isactivate(pos_idx, neg_idx, vlen=vlen, max_vlen=max_vlen)
    segment_list = get_segment(isactive)
    # print(segment_list)

    distance_score = np.zeros(max_vlen)
    for seg in segment_list:
        center = (seg[1] - seg[0]) / 2 + seg[0]
        width = seg[1] - seg[0] + 1
        sub_gauss =  center_width_gauss(center, width, vlen=vlen, max_vlen=max_vlen)
        distance_score[seg[0]:seg[1]+1] = sub_gauss[seg[0]:seg[1]+1]
    return distance_score



def get_distance_score_shift(pos_idx, neg_idx, vlen, max_vlen, shift):
    isactive = fill_isactivate(pos_idx, neg_idx, vlen=vlen, max_vlen=max_vlen)
    segment_list = get_segment(isactive)

    start_distance_score = np.zeros(max_vlen)
    for seg in segment_list:
        width = seg[1] - seg[0] + 1
        center = (seg[1] - seg[0]) / 2 + seg[0] - width * shift / 2
        sub_gauss =  center_width_gauss(center, width, vlen=vlen, max_vlen=max_vlen)
        start_distance_score[seg[0]:seg[1]+1] = sub_gauss[seg[0]:seg[1]+1]

    end_distance_score = np.zeros(max_vlen)
    for seg in segment_list:
        width = seg[1] - seg[0] + 1
        center = (seg[1] - seg[0]) / 2 + seg[0] + width * shift / 2
        sub_gauss =  center_width_gauss(center, width, vlen=vlen, max_vlen=max_vlen)
        end_distance_score[seg[0]:seg[1]+1] = sub_gauss[seg[0]:seg[1]+1]
    return start_distance_score, end_distance_score



def sigmoid(x):
    return 1/(1 + np.exp(-x))



def append_AP(p, active_point, gt_idx):
    gt_s, gt_e = gt_idx
    if gt_s <= p <= gt_e:
        active_point['pos_idx'].append(p)
    else:
        active_point['neg_idx'].append(p)
    return active_point




def get_uncert_model(prop_logits1, prop_logits2, vlen):
    Sprob1, Eprob1 = prop_logits1
    Sprob2, Eprob2 = prop_logits2

    Sprob1 = torch.sigmoid(torch.from_numpy(Sprob1))
    Sprob2 = torch.sigmoid(torch.from_numpy(Sprob2))
    Eprob1 = torch.sigmoid(torch.from_numpy(Eprob1))
    Eprob2 = torch.sigmoid(torch.from_numpy(Eprob2))
    
    Sprob1[vlen:] = 0
    Sprob2[vlen:] = 0
    Eprob1[vlen:] = 0
    Eprob2[vlen:] = 0


    s_frame_uncert = torch.abs(Sprob1 - Sprob2)
    e_frame_uncert = torch.abs(Eprob1 - Eprob2)
    return s_frame_uncert.numpy() + e_frame_uncert.numpy()

def infer_idx(start_prob, end_prob):
    start_prob = torch.from_numpy(start_prob)
    end_prob = torch.from_numpy(end_prob)
    outer = torch.matmul(start_prob.unsqueeze(1),end_prob.unsqueeze(0))
    outer = torch.triu(outer, diagonal=0)
    _, new_s_dix = torch.max(torch.max(outer, dim=1)[0], dim=0)  # (batch_size, )
    _, new_e_dix = torch.max(torch.max(outer, dim=0)[0], dim=0)  # (batch_size, )
    return new_s_dix.item(), new_e_dix.item()



def cp_testjson(gt_path, new_path):
    gt_test = os.path.join(os.path.split(gt_path)[0], "test.json")
    new_test = os.path.join(os.path.split(new_path)[0], "test.json")
    shutil.copy(gt_test, new_test)



def generate_configs(base_configs_path, task, I):
    conf = OmegaConf.load(base_configs_path)
    new_train_path = "./data/{}_re{}/train.json".format(task, I)
    new_test_path = "./data/{}_re{}/test.json".format(task, I)
    conf.paths.train_path = new_train_path
    conf.paths.test_path = new_test_path
    
    new_config_path = os.path.splitext(base_configs_path)
    new_config_path = new_config_path[0] + f"_re{I}" + new_config_path[1]
    with open(new_config_path, "w") as f:
        OmegaConf.save(conf, f)
    return new_config_path, conf






# def miou_two_dataset_idx(path1, path2, vlen_path):

#     with open(vlen_path, 'rb') as fp:
#         vlens = pickle.load(fp)
#     with open(path1, mode='r') as f:
#         data1 = json.load(f)
#     with open(path2, mode='r') as f:
#         data2 = json.load(f)
#     assert len(data1) == len(data2)
#     assert len(data1) == len(vlens)

#     miou = []
#     for i in range(len(data1)):
#         x1 = data1[i]
#         x2 = data2[i]
#         v  = vlens[i]
#         assert x1[0] == x2[0]
#         assert x1[0] == x2[0]
#         assert x1[0] == v['vid']
#         v_len = v["v_len"]
#         duration = x1[1]

#         x1_idx = time_to_index(x1[2], duration, v_len)
#         x2_idx = time_to_index(x2[2], duration, v_len)
#         iou = calculate_iou(x1_idx, x2_idx)
#         miou.append(iou)
#     return np.mean(miou)

# def miou_two_dataset_idx(path1, path2, vlen):
#     with open(path1, mode='r') as f:
#         data1 = json.load(f)
#     with open(path2, mode='r') as f:
#         data2 = json.load(f)
#     assert len(data1) == len(data2)

#     miou = []
#     for i in range(len(data1)):
#         x1 = data1[i]
#         x2 = data2[i]
#         assert x1[0] == x2[0]
#         assert x1[0] == x2[0]
#         duration = x1[1]
#         x1_idx = time_to_index(x1[2], duration, vlen)
#         x2_idx = time_to_index(x2[2], duration, vlen)
#         iou = calculate_iou(x1_idx, x2_idx)
#         miou.append(iou)
#     return np.mean(miou)



# def get_crossentroy(logist, label):
#     p = torch.from_numpy(logist).unsqueeze(0)
#     label = torch.tensor(label).unsqueeze(0)
#     return F.cross_entropy(p, label) 


# def get_klloss(prop_logits1, prop_logits2):
#     Sprob1, Eprob1 = prop_logits1
#     Sprob2, Eprob2 = prop_logits2
    
#     Sprob1 = torch.from_numpy(Sprob1)
#     Sprob2 = torch.from_numpy(Sprob2)
#     Eprob1 = torch.from_numpy(Eprob1)
#     Eprob2 = torch.from_numpy(Eprob2)
    
#     res = F.kl_div(Eprob1, Eprob2) + F.kl_div(Sprob1, Sprob2)
#     return res.item()


# def time_to_index(t, duration, vlen):
#     if isinstance(t, list):
#         res = []
#         for i in t:
#             res.append(time_to_index(i, duration, vlen))
#         return res
#     else:
#         return round(t / duration * (vlen - 1))

# def index_to_time(t, duration, vlen):
#     if isinstance(t, list):
#         res = []
#         for i in t:
#             res.append(index_to_time(i, duration, vlen))
#         return res
#     else:
#         return round(t / (vlen-1) * duration, 2)


# def hist_data(file_path):
#     with open(file_path, mode='r') as f:
#         data = json.load(f)
#     S, E = [], []
#     for i in data:
#         s, e = np.array(i[2]) / i[1]
#         S.append(s)
#         E.append(e)
#     return S, E




# def plt_hist(file_path, save_name, bins=64, high=20000):
#     S, E = hist_data(file_path)
#     plt.cla()
#     plt.hist(S, bins=bins)
#     plt.xlim(0, 1)
#     plt.ylim(0, high)
#     plt.savefig(save_name.format("S"))
#     plt.cla()
#     plt.hist(E, bins=bins)
#     plt.ylim(0, high)
#     plt.xlim(0, 1)
#     plt.savefig(save_name.format("E"))

