### Dynamic Entity Representations in Neural Language Models

This is a PyTorch implementation of the paper [Dynamic Entity Representations in Neural Language Models](https://www.google.com).  
We focus only on the Entity Prediction task, using the [InScript corpus](http://www.sfb1102.uni-saarland.de/?page_id=2582).

### Workflow
0. Install the requirements in 'requirements.txt'
1. First, we split the corpus into train/valid/test sets with './data/clean_data_split.json' and create the vocabulary dictionary with the train set.
```bash
python split_inscript.py --min_threshold 10
```
2.  Run training
```
python entitynlm.py
```


#### Example Usage
This command sets the embedding dim to 100, use pretrained Glove vectors, saves model at exp/demo and training history at runs/demo
```bash
python entitynlm.py --embed_dim 100 --pretrained --exp exp/demo --tensorboard runs/demo
```


