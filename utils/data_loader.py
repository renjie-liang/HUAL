import math
import random
import numpy as np
from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
from utils.data_utils import gene_soft_label
import matplotlib.pyplot as plt

class TrainLoader:
    def __init__(self, dataset, visual_features, configs):
        super(TrainLoader, self).__init__()
        self.dataset = dataset
        self.visual_feats = visual_features
        self.batch_size = configs.train.batch_size
        self.max_vlen = configs.model.max_vlen
        
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def num_samples(self):
        return len(self.dataset)

    def num_batches(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def batch_iter(self):
        random.shuffle(self.dataset)  # shuffle the train set first
        for index in range(0, len(self.dataset), self.batch_size):
            batch_data = self.dataset[index:(index + self.batch_size)]
            vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels = self.process_batch(batch_data)
            yield batch_data, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels

    def process_batch(self, batch_data):
        vfeats, word_ids, char_ids, s_inds, e_inds = [], [], [], [], []
        for data in batch_data:
            vfeat = self.visual_feats[data['vid']]
            vfeats.append(vfeat)
            word_ids.append(data['w_ids'])
            char_ids.append(data['c_ids'])
            s_inds.append(data['s_ind'])
            e_inds.append(data['e_ind'])
        batch_size = len(batch_data)
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
        # process video features
        vfeats, vfeat_lens = pad_video_seq(vfeats, max_length=self.max_vlen)
        # vfeats, vfeat_lens = pad_video_seq(vfeats)
        vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
        # process labels
        max_len = self.max_vlen
        s_labels = np.zeros(shape=[batch_size, max_len], dtype=np.float32)
        e_labels = np.zeros(shape=[batch_size, max_len], dtype=np.float32)
        match_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)  # (batch_size, v_seq_len)
        new_match_labels = np.zeros(shape=[batch_size, max_len, 4], dtype=np.float32)  # (batch_size, v_seq_len)
        new_s_labels, new_e_labels = [], []
        for idx in range(batch_size):
            st, et = s_inds[idx], e_inds[idx]
            cur_max_len = vfeat_lens[idx]
            # create classification labels
            s_labels[idx][0:cur_max_len] = 1e-10
            e_labels[idx][0:cur_max_len] = 1e-10
            y = (1 - cur_max_len * 1e-10 - 0.5) / 2
            s_labels[idx][st] = s_labels[idx][st] + 0.5
            if st > 0:
                s_labels[idx][st - 1] = y
            else:
                s_labels[idx][st] = s_labels[idx][st] + y
            if st < cur_max_len - 1:
                s_labels[idx][st + 1] = y
            else:
                s_labels[idx][st] = s_labels[idx][st] + y
            e_labels[idx][et] = e_labels[idx][et] + 0.5
            if et > 0:
                e_labels[idx][et - 1] = y
            else:
                e_labels[idx][et] = e_labels[idx][et] + y
            if et < cur_max_len - 1:
                e_labels[idx][et + 1] = y
            else:
                e_labels[idx][et] = e_labels[idx][et] + y
            # create matching labels
            ext_len = 2
            new_st_l = max(0, st - ext_len)
            new_st_r = min(st + ext_len, cur_max_len - 1)
            new_et_l = max(0, et - ext_len)
            new_et_r = min(et + ext_len, cur_max_len - 1)
            if new_st_r >= new_et_l:
                new_st_r = max(st, new_et_l - 1)
            match_labels[idx][new_st_l:(new_st_r + 1)] = 1  # add B-M labels
            match_labels[idx][(new_st_r + 1):new_et_l] = 2  # add I-M labels
            match_labels[idx][new_et_l:(new_et_r + 1)] = 3  # add E-M labels
            Ssoft, Esoft, _ = gene_soft_label(st, et, cur_max_len, max_len, 0.3)
            new_s_labels.append(Ssoft)
            new_e_labels.append(Esoft)
    
            new_match_labels[idx, np.where(match_labels[idx]==0), 0] = 1 
            new_match_labels[idx, np.where(match_labels[idx]==1), 1] = 1 
            new_match_labels[idx, np.where(match_labels[idx]==2), 2] = 1 
            new_match_labels[idx, np.where(match_labels[idx]==3), 3] = 1 
        return vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, new_match_labels
    
    
    # def process_batch(self, batch_data):
    #     vfeats, word_ids, char_ids, s_inds, e_inds = [], [], [], [], []
    #     for data in batch_data:
    #         vfeat = self.visual_feats[data['vid']]
    #         vfeats.append(vfeat)
    #         word_ids.append(data['w_ids'])
    #         char_ids.append(data['c_ids'])
    #         s_inds.append(data['s_ind'])
    #         e_inds.append(data['e_ind'])
    #     batch_size = len(batch_data)
    #     # process word ids
    #     word_ids, _ = pad_seq(word_ids)
    #     word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    #     # process char ids
    #     char_ids, _ = pad_char_seq(char_ids)
    #     char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    #     # process video features
    #     vfeats, vfeat_lens = pad_video_seq(vfeats, max_length=self.max_vlen)
    #     vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    #     vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    #     # process labels
    #     max_len = np.max(vfeat_lens)
    #     s_labels = np.zeros(shape=[batch_size, max_len], dtype=np.float32)
    #     e_labels = np.zeros(shape=[batch_size, max_len], dtype=np.float32)
    #     match_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)  # (batch_size, v_seq_len)
    #     new_match_labels = np.zeros(shape=[batch_size, max_len, 4], dtype=np.float32)  # (batch_size, v_seq_len)
    #     new_s_labels, new_e_labels = [], []
        
        
    #     for idx in range(batch_size):
    #         st, et = s_inds[idx], e_inds[idx]
    #         cur_max_len = vfeat_lens[idx]
    #         # create classification labels
    #         s_labels[idx][0:cur_max_len] = 1e-10
    #         e_labels[idx][0:cur_max_len] = 1e-10
    #         y = (1 - cur_max_len * 1e-10 - 0.5) / 2
    #         s_labels[idx][st] = s_labels[idx][st] + 0.5
    #         if st > 0:
    #             s_labels[idx][st - 1] = y
    #         else:
    #             s_labels[idx][st] = s_labels[idx][st] + y
    #         if st < cur_max_len - 1:
    #             s_labels[idx][st + 1] = y
    #         else:
    #             s_labels[idx][st] = s_labels[idx][st] + y
    #         e_labels[idx][et] = e_labels[idx][et] + 0.5
    #         if et > 0:
    #             e_labels[idx][et - 1] = y
    #         else:
    #             e_labels[idx][et] = e_labels[idx][et] + y
    #         if et < cur_max_len - 1:
    #             e_labels[idx][et + 1] = y
    #         else:
    #             e_labels[idx][et] = e_labels[idx][et] + y
    #         # create matching labels
    #         ext_len = 1
    #         new_st_l = max(0, st - ext_len)
    #         new_st_r = min(st + ext_len, cur_max_len - 1)
    #         new_et_l = max(0, et - ext_len)
    #         new_et_r = min(et + ext_len, cur_max_len - 1)
    #         if new_st_r >= new_et_l:
    #             new_st_r = max(st, new_et_l - 1)
    #         match_labels[idx][new_st_l:(new_st_r + 1)] = 1  # add B-M labels
    #         match_labels[idx][(new_st_r + 1):new_et_l] = 2  # add I-M labels
    #         match_labels[idx][new_et_l:(new_et_r + 1)] = 3  # add E-M labels
            

    #         new_match_labels[idx, np.where(match_labels[idx]==0), 0] = 1 
    #         new_match_labels[idx, np.where(match_labels[idx]==1), 1] = 1 
    #         new_match_labels[idx, np.where(match_labels[idx]==2), 2] = 1 
    #         new_match_labels[idx, np.where(match_labels[idx]==3), 3] = 1 

    #         Ssoft, Esoft, _ = gene_soft_label(st, et, cur_max_len, max_len, 0.3)
    #         new_s_labels.append(Ssoft)
    #         new_e_labels.append(Esoft)

    #     s_labels = np.stack(s_labels)
    #     e_labels = np.stack(e_labels)

    #     return vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, new_match_labels


class TestLoader:
    def __init__(self, datasets, visual_features, configs):
        self.visual_feats = visual_features
        self.val_set = None #if datasets['val_set'] is None else datasets['val_set']
        self.test_set = datasets
        self.batch_size = configs.train.batch_size
        self.max_vlen = configs.model.max_vlen

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def num_samples(self, mode='test'):
        if mode == 'val':
            if self.val_set is None:
                return 0
            return len(self.val_set)
        elif mode == 'test':
            return len(self.test_set)
        else:
            raise ValueError('Unknown mode!!! Only support [val | test | test_iid | test_ood].')

    def num_batches(self, mode='test'):
        if mode == 'val':
            if self.val_set is None:
                return 0
            return math.ceil(len(self.val_set) / self.batch_size)
        elif mode == 'test':
            return math.ceil(len(self.test_set) / self.batch_size)
        else:
            raise ValueError('Unknown mode!!! Only support [val | test].')

    def test_iter(self, mode='test'):
        if mode not in ['val', 'test']:
            raise ValueError('Unknown mode!!! Only support [val | test].')
        test_sets = {'val': self.val_set, 'test': self.test_set}
        dataset = test_sets[mode]
        if mode == 'val' and dataset is None:
            raise ValueError('val set is not available!!!')
        for index in range(0, len(dataset), self.batch_size):
            batch_data = dataset[index:(index + self.batch_size)]
            vfeats, vfeat_lens, word_ids, char_ids = self.process_batch(batch_data)
            yield batch_data, vfeats, vfeat_lens, word_ids, char_ids

    def process_batch(self, batch_data):
        vfeats, word_ids, char_ids, s_inds, e_inds = [], [], [], [], []
        for data in batch_data:
            vfeats.append(self.visual_feats[data['vid']])
            word_ids.append(data['w_ids'])
            char_ids.append(data['c_ids'])
            s_inds.append(data['s_ind'])
            e_inds.append(data['e_ind'])
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
        # process video features
        vfeats, vfeat_lens = pad_video_seq(vfeats, max_length=self.max_vlen)
        # vfeats, vfeat_lens = pad_video_seq(vfeats)
        vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
        return vfeats, vfeat_lens, word_ids, char_ids







class TrainNoSuffleLoader:
    def __init__(self, datasets, visual_features, configs):
        self.visual_feats = visual_features
        self.val_set = None
        self.test_set = datasets
        self.batch_size = configs.train.batch_size
        self.max_vlen = configs.model.max_vlen

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def num_samples(self, mode='test'):
        if mode == 'val':
            if self.val_set is None:
                return 0
            return len(self.val_set)
        elif mode == 'test':
            return len(self.test_set)
        else:
            raise ValueError('Unknown mode!!! Only support [val | test | test_iid | test_ood].')

    def num_batches(self, mode='test'):
        if mode == 'val':
            if self.val_set is None:
                return 0
            return math.ceil(len(self.val_set) / self.batch_size)
        elif mode == 'test':
            return math.ceil(len(self.test_set) / self.batch_size)
        else:
            raise ValueError('Unknown mode!!! Only support [val | test].')

    def test_iter(self, mode='test'):
        if mode not in ['val', 'test']:
            raise ValueError('Unknown mode!!! Only support [val | test].')
        test_sets = {'val': self.val_set, 'test': self.test_set}
        dataset = test_sets[mode]
        if mode == 'val' and dataset is None:
            raise ValueError('val set is not available!!!')
        for index in range(0, len(dataset), self.batch_size):
            batch_data = dataset[index:(index + self.batch_size)]
            vfeats, vfeat_lens, word_ids, char_ids = self.process_batch(batch_data)
            yield batch_data, vfeats, vfeat_lens, word_ids, char_ids

    def process_batch(self, batch_data):
        vfeats, word_ids, char_ids, s_inds, e_inds = [], [], [], [], []
        for data in batch_data:
            vfeats.append(self.visual_feats[data['vid']])
            word_ids.append(data['w_ids'])
            char_ids.append(data['c_ids'])
            s_inds.append(data['s_ind'])
            e_inds.append(data['e_ind'])
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
        # process video features
        vfeats, vfeat_lens = pad_video_seq(vfeats, max_length=self.max_vlen)
        vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
        vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
        return vfeats, vfeat_lens, word_ids, char_ids
