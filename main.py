import os, argparse, json
import tensorflow as tf
from tqdm import tqdm
from models.model import SeqPAN
from utils.data_gen import gen_or_load_dataset
from utils.data_loader import TrainLoader, TestLoader, TrainNoSuffleLoader
from utils.data_utils import load_json, load_yaml, save_json, load_video_features
from utils.runner_utils import eval_test_save, get_feed_dict, set_tf_config, train_epoch, test_epoch, get_logger
from datetime import datetime
from easydict import EasyDict


# parser = argparse.ArgumentParser(description='parameters settings')
# # data parameters
# parser.add_argument('--save_dir', type=str, default='datasets', help='path to save processed dataset')
# parser.add_argument('--task', type=str, default='activitynet', help='[activitynet | charades | charades_active | tacos]')
# parser.add_argument('--fv', type=str, default='i3d', help='features')
# parser.add_argument('--max_pos_len', type=int, default=100, help='maximal position sequence length allowed, '
#                                                                  'activitynet: 100, charades: 64, tacos: 256')
# # model parameters
# parser.add_argument('--num_words', type=int, default=None, help='word dictionary size')
# parser.add_argument('--num_chars', type=int, default=None, help='character dictionary size')
# parser.add_argument('--word_dim', type=int, default=300, help='word embedding dimension')
# parser.add_argument('--char_dim', type=int, default=100, help='character embedding dimension, activitynet: 100, '
#                                                               'charades/tacos: 50')
# parser.add_argument('--visual_dim', type=int, default=1024, help='video feature dimension [i3d: 1024 | c3d: 4096]')
# parser.add_argument('--dim', type=int, default=128, help='hidden size of the model')
# parser.add_argument('--num_heads', type=int, default=8, help='number of heads in transformer block')
# parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
# parser.add_argument('--attn_layer', type=int, default=2, help='number of attention layers')
# parser.add_argument('--match_lambda', type=float, default=1.0, help='weight of matching loss')
# parser.add_argument('--tau', type=float, default=0.3, help='temperature of gumbel softmax')
# parser.add_argument('--no_gumbel', action='store_true', help='whether use gumbel softmax')
# # training/evaluation parameters
# parser.add_argument('--gpu_idx', type=str, default='0', help='indicate which gpu is used')
# parser.add_argument('--seed', type=int, default=12345, help='random seed')
# parser.add_argument('--mode', type=str, default='train', help='[train | test]')
# parser.add_argument('--epochs', type=int, default=50, help='maximal training epochs')
# parser.add_argument('--num_train_steps', type=int, default=None, help='maximal training steps')
# parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
# parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
# parser.add_argument("--init_lr", type=float, default=0.0004, help="initial learning rate")
# parser.add_argument('--suffix', type=str, default="", help='saved checkpoint suffix')
# parser.add_argument('--resume', action='store_true')
# configs = parser.parse_args()
# print(configs)
# # set tensorflow configs

# # prepare or load dataset
# dataset = gen_or_load_dataset(configs)
# configs.num_chars = dataset['n_chars']
# configs.num_words = dataset['n_words']

# # get train and test loader
# print(configs.task)
# feature_path = os.path.join('data', 'features', configs.task)
# print(feature_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--debug', action='store_true', help='only debug')
    parser.add_argument('--re', type=str)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--seed', default=12345, type=int, help='random seed')
    parser.add_argument('--gpu_idx', type=str, default='0', help='indicate which gpu is used')
    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()

args = parse_args()
configs = EasyDict(load_yaml(args.config))
set_tf_config(args.seed, args.gpu_idx)

configs.paths.train_path = configs.paths.train_path.format(args.re)
configs.paths.test_path = configs.paths.test_path.format(args.re)
configs.re = args.re

dataset = gen_or_load_dataset(configs)
configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']

visual_features = load_video_features(configs.paths.feature_path, configs.model.max_vlen)
train_loader = TrainLoader(dataset=dataset['train_set'], visual_features=visual_features, configs=configs)
test_loader = TestLoader(datasets=dataset['test_set'], visual_features=visual_features, configs=configs)
train_nosuffle_loader = TrainNoSuffleLoader(datasets=dataset['train_set'], visual_features=visual_features, configs=configs)

model_dir = 'ckpt/{}'.format(configs.task)
os.makedirs(model_dir, exist_ok=True)
logger = get_logger("./logs/{}/{}".format(configs.task, args.suffix), configs.re)
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
                # r1i3, r1i5, r1i7, mi, loss_train = train_epoch(sess, train_loader, model, cur_lr, configs, get_feed_dict)
                # train_line = "TRAIN:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(r1i3, r1i5, r1i7, mi)
                # logger.info(train_line)

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
            
            

elif configs.mode.lower() in ['val', 'test']:
    if not os.path.exists(model_dir):
        raise ValueError('no pre-trained model exists!!!')
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # load model and test
    with tf.Graph().as_default() as graph:
        model = SeqPAN(configs=configs, graph=graph, word_vectors=dataset['word_vector'])
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=sess_config) as sess:
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            r1i3, r1i5, r1i7, mi, *_ = eval_test(sess=sess, model=model, data_loader=test_loader, mode=configs.mode)
            print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)


elif configs.mode.lower() == "eval_trainset":
    if not os.path.exists(model_dir):
        raise ValueError('no pre-trained model exists!!!')
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # load model and test
    with tf.Graph().as_default() as graph:
        model = SeqPAN(configs=configs, graph=graph, word_vectors=dataset['word_vector'])
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=sess_config) as sess:
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            r1i3, r1i5, r1i7, mi, *_ = eval_test_save(sess=sess, model=model, data_loader=train_nosuffle_loader, task=configs.task, suffix=configs.suffix, mode=configs.mode)
            print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)


else:
    raise ValueError("Unknown mode {}!!!".format(configs.mode))
