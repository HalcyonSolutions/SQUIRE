## SQUIRE: A Sequence-to-sequence Framework for Multi-hop Knowledge Graph Reasoning

### This code is based on the original project [SQUIRE](https://github.com/bys0318/SQUIRE) 

It is a <img src="figs/squire.gif" alt="drawing" width="40"/>framework for multi-hop reasoning, proposed in [SQUIRE: A Sequence-to-sequence Framework for Multi-hop Knowledge Graph Reasoning](https://arxiv.org/abs/2201.06206).

## Overview
We present **SQUIRE**, the first **S**e**q**uence-to-sequence based m**u**lt**i**-hop **re**asoning framework, which utilizes an encoder-decoder structure to translate the triple query to a multi-hop path. Here is an overview of our model architecture:

![](figs/model.png)

This is the PyTorch implementation of our proposed model.

## Training SQUIRE
To reproduce our results or extend SQUIRE model to more datasets, follow these steps.

#### Generate training set
First generate mapping files and query-path pairs as training set with `utils.py` under `data/` folder, run the following command:
```
python utils.py --dataset FB15K237 --gen-mapping --gen-eval-data --gen-train-data --num 6 --out 6_rev --max-len 3
```
To run our model on new datasets, it suffices to provide `train.txt`, `valid.txt`, `test.txt` files.

If using *rule-enhanced learning*, first generate mapping files by running:
```
python utils.py --dataset FB15K237 --gen-mapping --gen-eval-data
```
Then, our model utilizes [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/) to mine logical rules. We provide a convenient script `run.sh` under `AnyBURL/` for mining and filtering high confidence rules (please modify the dataset name in `run.sh` and `config-learn.properties`). The above step helps generate `rule.dict` containing high quality rules under the dataset folder, or alternatively you can use the `rule.dict` files we've already generated for you. Then go to `data/` folder and run:
```
python utils.py --dataset FB15K237 --gen-train-data --num 6 --out 6_rev_rule --max-len 3 --rule
```
Note that we are currently using BFS to search for query-path pairs in training set, which might take up to an hour on our experiment datasets. We are planning to optimize our code for speed-up.

#### Training and Evaluation
The following commands train and evaluate (on link prediction) SQUIRE model on all four datasets with GPU 0, where `--iter` is added to apply *iterative training* strategy during training. Check argparse configuration at `train.py` for details about each argument.
Remember to tune the vital hyperparameters, including `lr`, `num-epoch`, `label-smooth`, `prob` and `warmup`, so that SQUIRE can achieve promising performance on new datasets.

**FB15K237**
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset FB15K237 --embedding-dim 256 --hidden-size 512 \
    --num-layers 6 --batch-size 1024 --lr 5e-4 --dropout 0.1 --num-epoch 30 --save-dir "model_1" \ 
    --no-filter-gen --label-smooth 0.25 --encoder --save-interval 5 --l-punish --trainset "6_rev_rule" \ 
    --prob 0.15 --beam-size 256 --test-batch-size 8 --warmup 3 --iter
```

**NELL995**
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset NELL995 --embedding-dim 256 --hidden-size 512 \
    --num-layers 6 --batch-size 1024 --lr 1e-3 --dropout 0.1 --num-epoch 30 --save-dir "model_2" \
    --label-smooth 0.25 --encoder --save-interval 10 --l-punish --trainset "6_rev_rule" \
    --prob 0.15 --beam-size 512 --test-batch-size 2 --no-filter-gen --warmup 10 --iter --iter-batch-size 32
```

**FB15K237-20**
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset FB15K237-20 --embedding-dim 256 --hidden-size 512 \
    --num-layers 6 --batch-size 1024 --lr 1e-4 --dropout 0.1 --num-epoch 40 --save-dir "model_3" \
    --no-filter-gen --label-smooth 0.25 --encoder --save-interval 10 --l-punish --trainset "6_rev_rule" \
    --prob 0.25 --beam-size 256 --test-batch-size 4 --iter
```

**NELL23K**
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset NELL23K --embedding-dim 256 --hidden-size 512 \
    --num-layers 6 --batch-size 1024 --lr 5e-4 --dropout 0.1 --num-epoch 100 --save-dir "model_4" \
    --no-filter-gen --label-smooth 0.25 --encoder --save-interval 10 --l-punish --trainset "6_rev_rule" \
    --prob 0.15 --beam-size 512 --test-batch-size 4 --iter --iter-batch-size 32 
```

**Run SQUIRE training on question inputs**
```
CUDA_LAUNCH_BLOCKING=1 python train_question_input.py   --dataset mquake   --csv-path data/mquake/mquake_qa_[HOP#]hop.csv --entity2id data/mquake/entity2id.txt   --relation2id data/mquake/relation2id.txt   --question-tokenizer facebook/bart-base   --batch-size 16 --num-epoch 30 --beam-size 16 --max-len [HOP#: in case of nhop, please use 4]   --save-dir model_question_mquake_train_[HOP#]hop   --unfreeze-text-encoder --label-smooth 0.1   --l-punish --self-consistency --no-filter-gen  --test-batch-size 4 --embedding-dim [Embedding Dimension] [--tail-only: True, if you only care about last entity in the path] [--FULL: True trains on a FULL dataset, False trains on a Train dataset]
```

To evaluate a trained model (for example, on FB15K237), run the following command. To apply *self-consistency*, add `--self-consistency` command and keep `beam_size = 512`. Add `--output-path` command to observe the top generated correct path by SQUIRE. Remember to modify the --dataset to your desired test dataset name.
```
CUDA_VISIBLE_DEVICES=0 python train.py --test --dataset FB15K237 --beam-size 256 --save-dir "model_1" --ckpt "ckpt_30.pt" --test-batch-size 8 --encoder --l-punish --no-filter-gen
```

## Citation

Please cite our paper if you use our method in your work (Bibtex below).

```bibtex
@inproceedings{bai2022squire,
   title={SQUIRE: A Sequence-to-sequence Framework for Multi-hop Knowledge Graph Reasoning},
   author={Bai, Yushi and Lv, Xin and Li, Juanzi and Hou, Lei and Qu, Yincen and Dai, Zelin and Xiong, Feiyu},
   booktitle={EMNLP},
   year={2022}
}
```
### MQuAKE-ST
#### Summary Table (Hits@1)

| Model / QA Hop Size        | Graph-Type | 2-Hop   | 3-Hop   | 4-Hop   | n-Hop   |
| -------------------------- | ---------- | ------- | ------- | ------- | ------- |
| RW-End                     | Full       | 8.53e-3 | 3.61e-3 | 1.55e-3 | 3.39e-3 |
| RW-End                     | Train      | 8.53e-3 | 3.61e-3 | 1.56e-3 | 3.40e-3 |
| RW-Gold                    | Full       | 1.55e-3 | 6.69e-6 | 5.00e-8 | 3.26e-5 |
| **MINERVA ($d_{KG}$=100)** | Full       | 7.16e-1 | 3.62e-1 | 9.06e-1 | 8.16e-1 |
| **MINERVA ($d_{KG}$=100)** | Train      | 8.63e-1 | 8.54e-1 | 9.61e-1 | 8.65e-1 |
| MultiHopKG ($d_{KG}$=100)  | Full       | 4.33e-1 | 4.82e-1 | 3.55e-1 | 2.96e-1 |
| MultiHopKG ($d_{KG}$=100)  | Train      | 3.72e-1 | 3.51e-1 | 3.26e-1 | 3.47e-1 |
| **SQUIRE ($d_{KG}$=100)**  | Full       | 8.38e-1 | 8.68e-1 | 6.64e-1 | 8.35e-1 |
| **SQUIRE ($d_{KG}$=100)**  | Train      | 4.99e-1 | 8.15e-1 | 6.74e-1 | 6.75e-1 |

---

### Test Results for SQUIRE ($d_{KG}$=100) on Full Graph

| QA & Reasoning | Hits@1  | Hits@3  | Hits@5  | Hits@10     | Hits@20 | MRR         |
| -------------- | ------- | ------- | ------- | ----------- | ------- | ----------- |
| **2-Hop**      | 8.75e-1 | 9.42e-1 | 9.62e-1 | 9.94e-1 | 9.99e-1 | 9.16e-1 | 
| **3-Hop**      | 9.18e-1 | 9.42e-1 | 9.49e-1 | 9.51e-1 | 9.52e-1 | 9.30e-1 |
| **4-Hop**      | 9.42e-1 | 9.60e-1 | 9.63e-1 | 9.63e-1 | 9.64e-1 | 9.51e-1 |
| **n-Hop**      | 8.80e-1 | 8.62e-1 | 8.80e-1 | 8.89e-1 | 8.90e-1 | 8.35e-1 |

---

### Test Results for SQUIRE ($d_{KG}$=100) on Train Graph

| QA & Reasoning | Hits@1  | Hits@3  | Hits@5  | Hits@10     | Hits@20 | MRR         |
| -------------- | ------- | ------- | ------- | ----------- | ------- | ----------- |
| **2-Hop**      | 4.68e-1 | 6.21e-1 | 6.64e-1 | 7.09e-1 | 7.13e-1 | 5.53e-1 |
| **3-Hop**      | 5.35e-1 | 7.67e-1 | 7.83e-1 | 8.11e-1 | 8.11e-1 | 6.44e-1 |
| **4-Hop**      | 7.95e-1 | 9.33e-1 | 9.44e-1 | 9.44e-1 | 9.50e-1 | 8.63e-1 |
| **n-Hop**      | 6.38e-1 | 7.33e-1 | 7.56e-1 | 7.79e-1 | 7.82e-1 | 6.90e-1 |
