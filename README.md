# Curriculum-Based Self-Training Makes Better Few-Shot Learners for Data-to-Text Generation

## Introduction

Curriculum-Based Self-Training (CBST) utilizes curriculum learning to construct pseudo-labeled data from easy cases to hard ones, and leverages such data into the self-training process at different iterations. You can read our [paper](https://www.ijcai.org/proceedings/2022/0580) for more details. This project is a PyTorch implementation of our work.

## Dependencies

* Python 3.7
* NumPy
* PyTorch 1.4.0
* Transformers (Huggingface) 3.0.0

## Quick Start

**NOTE**: In order to compute the METEOR scores, please download the required [data](https://github.com/wenhuchen/Data-to-text-Evaluation-Metric/blob/master/pycocoevalcap/meteor/data/paraphrase-en.gz) and put it under the following folder: `pycocoevalcap/meteor/data/`.

### Datasets

Our experiments involve two datasets, i.e., WebNLG and WikiBio. The raw data are from the GitHub repository of [KGPT](https://github.com/wenhuchen/KGPT). You can download the pre-processed datasets used in our paper on [Google Drive](https://drive.google.com/drive/folders/1HD2DilvGj0wK4hMx1UpAiNvuwjjnJ5XS?usp=sharing) / [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/09464df29b004159b2c4/). This data folder also contains a file `counterfitted_neighbors.json` used for word substitution, which originates from the GitHub repository of [certified-word-sub](https://github.com/robinjia/certified-word-sub).

### Training

You can download the model checkpoint of [BART](https://huggingface.co/facebook/bart-base) provided by Huggingface Transformers, and train the model on two datasets, respectively.

```shell
bash finetune_bart_webnlg.sh
bash finetune_bart_wikibio.sh
```

In the scripts, `--output_dir` denotes the directory to save the intermediate and final models. `--model_path` indicates the pre-trained checkpoint used for initialization.  `--model_path` and `--tokenizer_path` are set to the directory of the downloaded BART checkpoint. You can refer to the codes for the details of other hyper-parameters.

### Inference

We also provide the inference scripts to directly acquire the generation results on the test sets.

```shell
bash infer_bart_webnlg.sh
bash infer_bart_wikibio.sh
```

In the scripts, `--output_dir` denotes the directory of model checkpoints used for inference. The generated results are also saved in this directory.

## Citation

```
@inproceedings{ke2022cbst,
  title     = {Curriculum-Based Self-Training Makes Better Few-Shot Learners for Data-to-Text Generation},
  author    = {Ke, Pei and Ji, Haozhe and Yang, Zhenyu and Huang, Yi and Feng, Junlan and Zhu, Xiaoyan and Huang, Minlie},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {4178--4184},
  year      = {2022},
}
```

Please kindly cite our paper if this paper and the codes are helpful.

## Thanks

Many thanks to the GitHub repositories of [Transformers](https://github.com/huggingface/transformers) and [bart-closed-book-qa](https://github.com/shmsw25/bart-closed-book-qa). Part of our codes are modified based on their codes.
