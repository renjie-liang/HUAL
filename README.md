# HUAL
CVPR 2023 "Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning"

** There are still some bugs when inferring, because we reconstruct our code for easily reading. **

![overview](/images/architecture.png)

## Prerequisites
- python3 with tensorflow>=`2.0`, pytorch, tqdm, nltk, numpy, eta.

## Preparation
The visual features of `Charades-STA` and `ActivityNet Captions` are available at [Box Drive](
https://app.box.com/s/d7q5atlidb31cuj1u8znd7prgrck1r1s), download and place them under the `./data/features/` directory. 
Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it to 
`./data/features/` directory. Directory hierarchies are shown below:
```
HUAL
    |____ ckpt/
    |____ data/
        |____ datasets/
        |____ features/
            |____ activitynet/
            |____ charades/
            |____ glove/glove.840B.300d.txt

    |____ scripts_iter/
        |____ data/
            |___PSEUDO_LABEL/
            
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
