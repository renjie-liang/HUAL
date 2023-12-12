import os
import codecs
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from utils.data_utils import load_json, load_lines, load_pickle, save_pickle, time_to_index

PAD, UNK = "<PAD>", "<UNK>"

class Processor:
    def __init__(self):
        super(Processor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data(self, data, scope):
        results = []
        for record in tqdm(data, total=len(data), desc='process data {}'.format(scope)):
            vid, duration, gt_label, sentence = record[:4]
            start_time, end_time = gt_label
            words = word_tokenize(sentence.strip().lower(), language="english")

            record = {'sample_id': self.idx_counter, 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                    'duration': duration, 'words': words}
            results.append(record)
            self.idx_counter += 1
        return results

    def convert(self, data_dir):
        self.reset_idx_counter()
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))
        # load raw data
        train_data = load_json(os.path.join(data_dir, 'train.json'))
        test_data = load_json(os.path.join(data_dir, 'test.json'))

        # process data
        train_set = self.process_data(train_data, scope='train')
        test_set = self.process_data(test_data, scope='test')
        return train_set, None, test_set  # train/val/test



def load_glove(glove_path):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict, glove_path):
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)
    return np.asarray(vectors)


def vocab_emb_gen(datasets, emb_path):
    # generate word dict and vectors
    emb_vocab = load_glove(emb_path)
    word_counter, char_counter = Counter(), Counter()
    for data in datasets:
        for record in data:
            for word in record['words']:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict, emb_path)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 5]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors


def dataset_gen(data, vfeat_lens, word_dict, char_dict, max_pos_len, scope):
    dataset = list()
    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            continue
        s_ind, e_ind = time_to_index(record['s_time'], record['e_time'], vfeat_lens[vid], record['duration'])
        word_ids, char_ids = [], []
        for word in record['words'][0:max_pos_len]:
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)
        result = {'sample_id': record['sample_id'], 'vid': record['vid'], 's_time': record['s_time'],
                  'e_time': record['e_time'], 'duration': record['duration'], 'words': record['words'],
                  's_ind': int(s_ind), 'e_ind': int(e_ind), 'v_len': vfeat_lens[vid], 'w_ids': word_ids,
                  'c_ids': char_ids}
        dataset.append(result)
    return dataset


def dataset_gen_active(data, vfeat_lens, word_dict, char_dict, max_pos_len, scope):
    def active_weight_to_index(active_weight, flen):
        active_weight = np.array(active_weight)
        index_list = (active_weight >= 0.5)
        tmp = np.where(index_list)[0]
        # if len(tmp) == 1:
        #     tmp = np.append(tmp, [tmp[0]+1])
        if len(tmp) < 1:
            print(tmp)
            print(active_weight)
            raise
        s_idx = round(tmp[0] / len(active_weight-1) * (flen-1))
        e_idx = round(tmp[-1] / len(active_weight-1) * (flen-1))
        return s_idx, e_idx

    dataset = list()
    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            continue
        s_ind, e_ind = active_weight_to_index(record['active_weight'], vfeat_lens[vid])

        word_ids, char_ids = [], []
        for word in record['words'][0:max_pos_len]:
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)
        result = {'sample_id': record['sample_id'], 'vid': record['vid'], 's_time': record['s_time'],
                  'e_time': record['e_time'], 'duration': record['duration'], 'words': record['words'],
                  's_ind': int(s_ind), 'e_ind': int(e_ind), 'v_len': vfeat_lens[vid], 'w_ids': word_ids,
                  'c_ids': char_ids}
        dataset.append(result)
    return dataset

def gen_train_data_cache_path(configs):
    feat_version = os.path.split(configs.paths.feature_path)[-1]
    save_path = os.path.join(configs.paths.cache_dir, '_'.join([configs.task, feat_version, str(configs.model.max_vlen), configs.suffix]) + '.pkl')
    return save_path


def gen_or_load_dataset(configs):
    if not os.path.exists(configs.paths.cache_dir):
        os.makedirs(configs.paths.cache_dir)
    data_dir = os.path.join('data', configs.task + "_" + configs.suffix)
    print("--"*20)
    print(data_dir)
    print("--"*20)

    save_path = gen_train_data_cache_path(configs)
    if os.path.exists(save_path):
        dataset = load_pickle(save_path)
        return dataset
    feat_len_path = os.path.join(configs.paths.feature_path, 'feature_shapes.json')
    
    # load video feature length
    vfeat_lens = load_json(feat_len_path)
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(configs.model.max_vlen, vfeat_len)
        
    # load data
    processor = Processor()
    train_data, val_data, test_data = processor.convert(data_dir)
    # generate dataset
    data_list = [train_data, test_data] if val_data is None else [train_data, val_data, test_data]
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, configs.paths.glove_path)
    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'train')
    # train_set = dataset_gen_active(train_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'train')
    val_set = None if val_data is None else dataset_gen(val_data, vfeat_lens, word_dict, char_dict,
                                                        configs.max_pos_len, 'val')
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'test')
    # save dataset
    n_val = 0 if val_set is None else len(val_set)
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': n_val,
               'n_test': len(test_set), 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, save_path)
    return dataset
