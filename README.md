## CosRec: 2D Convolutional Neural Networks for Sequential Recommendation

This is our PyTorch implementation for the paper:

*CosRec: 2D Convolutional Neural Networks for Sequential Recommendation, CIKM-2019*

[[arXiv](https://arxiv.org/abs/1908.09972)] [[GitHub](https://github.com/zzxslp/CosRec)]

The code is tested on a Linux server (w/ NVIDIA GeForce Titan X Pascal) with PyTorch 1.1.0 and Python 3.7.

## Requirements
* Python 3
* PyTorch v1.0+ (v0.4+ might also work)

## Training
To train our model on `ml1m` (with default hyper-parameters): 

```
python train.py --dataset=ml1m
```

or on `gowalla` (change a few hyper-paras based on dataset statistics):

```
python train.py --dataset=gowalla --d=100 --fc_dim=50 --l2=1e-6
```

You should be able to obtain MAPs of ~0.188 and ~0.098 on ML-1M and Gowalla respectively, with the above settings.

## Datasets

- Datasets are organized into 2 separate files: **_train.txt_** and **_test.txt_**

- Same as other data format for recommendation, each file contains a collection of triplets:

  > user item rating

  The only difference is the triplets are organized in *time order*.

- As the problem is Sequential Recommendation, the rating doesn't matter, so we convert them all to 1.

## Citation

If you find this repository useful, please cite our paper:

```
@inproceedings{yan2019cosrec,
  title={CosRec: 2D Convolutional Neural Networks for Sequential Recommendation},
  author={Yan, An and Cheng, Shuo and Kang, Wang-Cheng and Wan, Mengting and McAuley, Julian},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={2173--2176},
  year={2019},
  organization={ACM}
}
```

## Acknowledgments

This project is built on top of [Spotlight](https://github.com/maciejkula/spotlight) and [Caser](https://github.com/graytowne/caser_pytorch). Thanks [Maciej](https://github.com/maciejkula) and [Jiaxi](https://github.com/graytowne) for their contributions to the community.