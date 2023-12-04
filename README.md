# HUAL
CVPR 2023 "Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning"

** There are still some bugs when inferring, because we reconstruct our code for easily reading. **

![overview](/images/architecture.png)

## Prerequisites
- pip install 

## Preparation
The visual features of `Charades-STA` and `ActivityNet Captions` are available at [Box Drive](
https://app.box.com/s/d7q5atlidb31cuj1u8znd7prgrck1r1s).
Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip). Modify the path setting in `./configs/charades/SeqPAN.yaml`.

Directory hierarchies are shown below:
```
HUAL
    |____ ckpt/
    |____ logs/
    |____ data_pkl/
    |____ data/
        |____ anet_gt/
        |____ anet_re0/
        |____ anet_re1/
        ...
        |____ anet_re10/

    |____ update_label/
        ...
    |___ results/
        |____ activitynet/
        |____ charades/
    ...
```

## Quick Start
**Train**
```shell script
# The above scripts can both update pseudo label and train model iteratively.
# The processing can also be run step by step as follow. '--re' is the times of iteration. 
# 1. Update pseudo label.
python update_label/update_charades.py 0
# 2. Train model.
python main.py --config ./configs/charades/SeqPAN.yaml --gpu_idx 1 --re 0 --suffix debug --mode train
# 3. Generate model predict of train dataset.
python main.py --config ./configs/charades/SeqPAN.yaml --gpu_idx 1 --re 0 --suffix debug --mode test

```
