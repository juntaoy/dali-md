# Neural Mention Detection

## Introduction
This repository contains code introduced in the following paper:
 
**[Neural Mention Detection](https://arxiv.org/abs/1907.12524)**  
Juntao Yu, Bernd Bohnet and Massimo Poesio  
In *Proceedings of the 12th Language Resources and Evaluation Conference (LREC)*, 2020

## Setup Environments
* The code is written in Python 2, the compatibility to Python 3 is not guaranteed.  
* Before starting, you need to install all the required packages listed in the requirment.txt using `pip install -r requirements.txt`.
* After that run `setup.sh` to compile the Tensorflow custom kernels.
* If you want to use `Lee MD` you need to uncomment the first few lines of `setup.sh` to download the GloVe embeddings that required by the system.

## To use a pre-trained model
* Pre-trained models can be download from [this link](https://www.dropbox.com/s/pnidyrrv33mbj3z/best_models.zip?dl=0). We provide two pre-trained models trained with our best model (`Biaffine MD`):
   * One trained on the [CoNLL 2012 shared task](http://conll.cemantix.org/2012/introduction.html) data in which singletons and non-referring expressions are not annotated.
   * The other trained on the [CRAC 2018 shared task](http://dali.eecs.qmul.ac.uk/crac18_shared_task) data that has both single mentions and the non-referring expressions annotated. 
* Choose the model you want to use and put the `model.max.ckpt.*` files under the `logs/biaffinmd` folder.
* Modifiy the *test_path* accordingly:
   * the *test_path* is the path to *.jsonlines* file, each line of the *.jsonlines* file must in the following format:
   
   ```
  {
  "clusters": [[[0,0],[5,5]],[[2,3],[7,8]], 
  "doc_key": "nw",
  "sentences": [["John", "has", "a", "car", "."], ["He", "washed", "the", "car", "yesteday","."],["Really","?","it", "was", "raining","yesteday","!"]],
  
  }
  ```
  
  * If you only have mentions annotated, but not the coreference clusters, then you can simply give every mention a cluster, the reason we use `clusters` instead of `mentions` is to allow the same data also be used by our coreference resolution system.
* The model has two output mode `high-recall` and `high-f1` which can be configured in the `experiments.conf`
* Then use `python evaluate.py config_name` to start your evaluation

## To train your own model
* If you plan to use `LEE MD` or `Biaffine MD`, you need to run the `python cache_elmo.py train.jsonlines dev.jsonlines` to store the ELMo embeddings in the disk, this will speed up your training a lot.
* If you plan to use `LEE MD` you will additionally need to create the character vocabulary by using `python get_char_vocab.py train.jsonlines dev.jsonlines`
* If you plan to use `Bert MD` we would suggest you train on the CPU instead unless you have a very small data set, see [BERT page](https://github.com/google-research/bert) for more information about the GPU memory issue.
* Finally you can start training by using `python train.py config_name`

## Training speed
Both `LEE MD` and `Biaffine MD` can be trained in just a few (4-6) hours on a GTX 1080Ti GPU. The `Bert MD` however takes about 1 week to finish on 48 CPU cores.
