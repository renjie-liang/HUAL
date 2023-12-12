import os, json
import argparse
import tensorflow as tf
from tqdm import tqdm
from models.model import SeqPAN
from utils.data_gen import gen_or_load_dataset
from utils.data_loader import TrainLoader, TestLoader, TrainNoSuffleLoader
from utils.data_utils import load_json, save_json, load_video_features, load_yaml
from utils.runner_utils import eval_test_save, get_feed_dict, get_logger, set_tf_config, test_epoch, train_epoch
from datetime import datetime

from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--debug', action='store_true', help='only debug')
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--seed', default=12345, type=int, help='random seed')
    parser.add_argument('--gpu_idx', type=str, default='0', help='indicate which gpu is used')
    parser.add_argument('--ckpt_dir', type=str, default='')
    
    return parser.parse_args()

args = parse_args()
configs = EasyDict(load_yaml(args.config))
set_tf_config(args.seed, args.gpu_idx)
configs.suffix = args.suffix

# prepare or load dataset
dataset = gen_or_load_dataset(configs)
configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']
visual_features = load_video_features(configs.paths.feature_path, configs.model.max_vlen)
train_loader = TrainLoader(dataset=dataset['train_set'], visual_features=visual_features, configs=configs)
# test_loader = TestLoader(datasets=dataset['test_set'], visual_features=visual_features, configs=configs)
test_loader = TestLoader(datasets=dataset, visual_features=visual_features, configs=configs)
train_nosuffle_loader = TrainNoSuffleLoader(datasets=dataset['train_set'], visual_features=visual_features, configs=configs)

model_dir = 'ckpt/{}_'.format(configs.task, configs.suffix)
os.makedirs(model_dir, exist_ok=True)
logger = get_logger(f"./logs/{configs.task}", args.suffix)
logger.info(json.dumps(configs, indent=4))




if args.mode.lower() == 'train':
    with tf.Graph().as_default() as graph:
        model = eval(configs.model.name)(configs=configs, graph=graph, word_vectors=dataset['word_vector'])
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        best_r1i7, best_r1i7_performance = -1.0, None
        with tf.compat.v1.Session(config=sess_config) as sess:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(configs.train.epochs):
                logger.info("Epoch {}|{}:".format(epoch, configs.train.epochs))
                cur_lr = configs.train.lr * (1.0 - epoch / configs.train.epochs)
                r1i3, r1i5, r1i7, mi, loss_train = train_epoch(sess, train_loader, model, cur_lr, configs, get_feed_dict)
                train_line = "TRAIN:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(r1i3, r1i5, r1i7, mi)
                logger.info(train_line)

                r1i3, r1i5, r1i7, mi = test_epoch(sess, model, test_loader)
                test_line = "TEST:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(r1i3, r1i5, r1i7, mi)
                logger.info(test_line)
                
                ## save the model according to the result of Rank@1, IoU=0.7
                if r1i7 > best_r1i7:
                    best_r1i7 = r1i7
                    filename = os.path.join(model_dir, "best_{}.ckpt".format(configs.model.name))
                    saver.save(sess, filename)
                    best_r1i7_performance = "\n" + train_line + "\n" + test_line

            logger.info("\n\nHighest R1i7 epoch\n")
            logger.info(best_r1i7_performance)



if args.mode.lower() == 'test':
    if not os.path.exists(model_dir):
        raise ValueError('no pre-trained model exists!!!')
    with tf.Graph().as_default() as graph:
        model = eval(configs.model.name)(configs=configs, graph=graph, word_vectors=dataset['word_vector'])
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        best_r1i7, best_r1i7_performance = -1.0, None
        with tf.compat.v1.Session(config=sess_config) as sess:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            r1i3, r1i5, r1i7, mi = test_epoch(sess, model, test_loader)
            test_line = "TEST:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(r1i3, r1i5, r1i7, mi)
            logger.info(test_line)
            

elif args.mode.lower() == "infer_trainset":
    if not os.path.exists(model_dir):
        raise ValueError('no pre-trained model exists!!!')
    with tf.Graph().as_default() as graph:
        model = SeqPAN(configs=configs, graph=graph, word_vectors=dataset['word_vector'])
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=sess_config) as sess:
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            r1i3, r1i5, r1i7, mi, *_ = eval_test_save(sess=sess, model=model, data_loader=train_nosuffle_loader, task=configs.task, suffix=configs.suffix)
            test_line = "predict train set:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(r1i3, r1i5, r1i7, mi)
else:
    raise ValueError("Unknown mode {}!!!".format(configs.mode))
