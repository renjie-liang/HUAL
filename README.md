# HUAL
<<<<<<< HEAD
CVPR 2023 "Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning"


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

# iterative train ActivityNet Captions dataset
python run_anet.py
# iterative train Charades-STA dataset
python run_charades.py

# The above scripts can both update pseudo label and train model iteratively.
# The processing can also be run step by step as follow. 'suffix' is the name of iteration, 'I' is the times of iteration. 
# 1. Update pseudo label.
python ./scripts_iter/iter_charades_TTT.py suffix I
# 2. Train model.
python main.py --task charades --max_pos_len 64 --char_dim 50  --suffix 'suffix+I'
# 3. Generate model predict of train dataset.
python main.py --task charades --max_pos_len 64 --char_dim 50  --suffix 'suffix+I'

```
**Test**
```shell script

# Test ActivityNet Captions dataset
python main.py --task activitynet --max_pos_len 100 --char_dim 100 --suffix 'suffix+I' --mode test

# Test Charades-STA dataset
python main.py --task charades --max_pos_len 64 --char_dim 50 --suffix 'suffix+I' --mode test

# For convenience, the follow script can summary perfomacnce.
python summary_performance.py suffix
```





>>>>>>> 885ba71... init
=======
CVPR 2023 "Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning"
=======
CVPR 2023 "Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning" 


>>>>>>> f7b77ee... Update README.md
Coming soon...
>>>>>>> 858ff0f... Update README.md
