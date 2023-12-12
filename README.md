# HUAL
CVPR 2023 "Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning"


![overview](/images/architecture.png)

## Prerequisites
- pip install 

## Preparation
The visual features of `Charades-STA` and `ActivityNet Captions` are available at [Box Drive](
https://app.box.com/s/d7q5atlidb31cuj1u8znd7prgrck1r1s).

Download the word embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip). Modify the path setting in `./configs/charades/SeqPAN.yaml`.

Download the anet initiate pkl from [Box Drive](https://app.box.com/s/mlcxc8oq3zanz2mzamze0hb9yyt51ij8). , put it to ./results/anet/re0.pkl

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
    |___ results/
        |____ anet/re0.pkl
        |____ charades/re0.pkl
    ...
```

## Quick Start
**Train**
```shell script
python run_charades.py
python run_anet.py

# it include three step:
# 1. Update pseudo label.
# 2. Train model.
# 3. Infer prediction of train dataset.

```


### Citation
If you feel this project is helpful to your research, please cite our work.
```
@InProceedings{Ji_2023_CVPR,
    author    = {Ji, Wei and Liang, Renjie and Zheng, Zhedong and Zhang, Wenqiao and Zhang, Shengyu and Li, Juncheng and Li, Mengze and Chua, Tat-seng},
    title     = {Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-Based Active Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {23013-23022}
}
```