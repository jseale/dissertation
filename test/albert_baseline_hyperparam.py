import os
import sys
import logging
import re
import random
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
from typing import List, Dict, Tuple, TextIO
from transformers import AlbertForTokenClassification
from transformers import AlbertTokenizerFast, EvalPrediction
from transformers import Trainer, TrainingArguments
from transformers import AdamW
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import ray
from numba import jit, cuda
ray.init(object_store_memory=15000000000, include_dashboard=False) 

WHD_BERT_DATA = "C:/Users/jseal/Dev/dissertation/Data/WHD_ALBERT/variant_stratified/"
WHD = 'C:/Users/jseal/Dev/dissertation/Data/WikipediaHomographData/data/'

#Txt splits
TRAIN_TXT = WHD_BERT_DATA + "train.txt"
DEV_TXT = WHD_BERT_DATA + "dev.txt"
TEST_TXT = WHD_BERT_DATA + "test.txt"

#Logging path
LOGS = './whd-variant-stratified-albert-model_logs_hyperparam_search'

#Output dir
OUT_DIR = './whd-variant-stratified-albert-model-hyperparam'

#Labels file
LABELS = WHD_BERT_DATA + "labels.txt"

#Homographs
#whd_df = pd.read_csv(WHD + 'WikipediaHomographData.csv')
#homographs = whd_df.drop_duplicates(subset='homograph')['homograph'].tolist()

# Model Variables
MAX_LENGTH = 128 #@param {type: "integer"}
BATCH_SIZE = 16 #@param {type: "integer"}
NUM_EPOCHS = 10 #@param {type: "integer"}
SAVE_STEPS = 100 #@param {type: "integer"}
LOGGING_STEPS = 100 #@param {type: "integer"}
#SEED = random.randrange(10000)
SEED_1 = random.randrange(10000)
print(SEED_1)
SEED_2 = random.randrange(10000)
print(SEED_2)
SEED_3 = random.randrange(10000)
print(SEED_3)
SEED_4 = random.randrange(10000)
print(SEED_4)
STEPS = "steps"
EVAL_STEPS = 100
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

MODEL_NAME = "albert-base-v2"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

logger = logging.getLogger(__name__)


def read_set(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text(encoding="utf8").strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            if token.isalpha():
                tokens.append(token)
                tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def encode_tags(tags, encodings): 
	labels = [[tag2id[tag] for tag in doc] for doc in tags] 
	encoded_labels = []
	for doc_labels, doc_offset, e in zip(labels, encodings.offset_mapping, encodings.encodings): 
		indices = []
		for i, tok in enumerate(e.tokens): 
			if len(tok) == 1: 
				if (tok == '_'): 
					indices.append(i)
		tokens = [v for i, v in enumerate(e.tokens) if i not in indices]
		doc_offset = [v for i, v in enumerate(doc_offset) if i not in indices] 
	for i in range(len(indices)): 
		tokens.insert(-1, (0,0))
	doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
	arr_offset = np.array(doc_offset)

	doc_enc_labels[(arr_offset[:,0]==0) & (arr_offset[:,1] != 0)] = doc_labels
	encoded_labels.append(doc_enc_labels.tolist())

	return encoded_labels


class WHDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
       # self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def align_predictions(predictions: np.ndarray, 
                      label_ids: np.ndarray) -> Tuple[List[int], List[int], List[str]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(id2tag[label_ids[i][j]])
                preds_list[i].append(id2tag[preds[i][j]])
    
    return preds_list, out_label_list

def homograph_check(preds_list, out_label_list):
    '''Check that predictions to be evaluated are homographs, 
    not tokens from the sentence provided as context.'''
    pred_labels = zip(preds_list, out_label_list)
    unique_tags = [label.strip("\n") for label in open(LABELS).readlines()]
    preds = []
    labels = []
    for e in pred_labels:
        pred_label_pair = zip(e[0], e[1])
        for pred, label in pred_label_pair:
            if label is not 'O':
                preds.append(pred)
                labels.append(label)
    return labels, preds
    
def compute_metrics(p: EvalPrediction) -> Dict:
    '''Only compute metrics on homograph pronunciation predictions'''
    preds_list, labels_list = align_predictions(p.predictions, p.label_ids)
    targets, preds = homograph_check(preds_list, labels_list)
#     preds = [tag2id[pred] for pred in preds]
#     targets = [tag2id[target] for target in targets]
#     preds = torch.as_tensor(preds)
#     targets = torch.as_tensor(targets)
    return {
        "accuracy_score": accuracy_score(targets, preds),
        "balanced_accuracy_score": balanced_accuracy_score(targets, preds),
        #"micro_auc_roc": roc_auc_score(targets, preds, average='micro')
#         "micro_accuracy": accuracy(preds, targets, class_reduction='micro', num_classes=len(unique_tags)),
#         "macro_accuracy": accuracy(preds, targets, class_reduction='macro', num_classes=len(unique_tags)),
    }


if __name__ == '__main__':
    #Get unique labels, enumeration/IDs, and make mappings
    unique_tags = [label.strip("\n") for label in open(LABELS).readlines()]
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    #Make separate lists of aligned texts and tags
    train_texts, train_tags = read_set(TRAIN_TXT)
    dev_texts, dev_tags = read_set(DEV_TXT)
    test_texts, test_tags = read_set(TEST_TXT)
    #Make encodings for text 
    tokenizer = AlbertTokenizerFast.from_pretrained(MODEL_NAME,add_prefix_space=True)
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    dev_encodings = tokenizer(dev_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

     #Make encodings for labels
    train_labels = encode_tags(train_tags, train_encodings)
    dev_labels = encode_tags(dev_tags, dev_encodings)

    #Make datasets
    train_encodings.pop("offset_mapping") # don't want to pass this to the model
    dev_encodings.pop("offset_mapping")
    train_dataset = WHDataset(train_encodings, train_labels)
    dev_dataset = WHDataset(dev_encodings, dev_labels)
    #print(dev_dataset[0])

     #Instantiate model, training args and trainer; train and evaluate
    def model_init(): 
        model = AlbertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_tags))
        return model.to(device)

    training_args = TrainingArguments(
        load_best_model_at_end=True,
        output_dir=OUT_DIR,    # output directory
        num_train_epochs=NUM_EPOCHS,              # total # of training epochs
        per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
        warmup_steps=WARMUP_STEPS,                # number of warmup steps for learning rate scheduler
        weight_decay=WEIGHT_DECAY,                # strength of weight decay
        logging_dir=LOGS,                         # directory for storing logs
        evaluation_strategy=STEPS,
        eval_steps=EVAL_STEPS,
        seed=SEED_1,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model_init=model_init,         # instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,            # evaluation dataset
        compute_metrics=compute_metrics
    )

    # trainer.train()
    # trainer.evaluate()
    @jit(target="cuda")
    def get_best_trial(trainer): 
        best_trial = trainer.hyperparameter_search(
            direction="maximize", 
            backend="ray", 
            n_trials=10, # number of trials
            n_jobs=0
        )
        return best_trial

    best_trial = get_best_trial(trainer)

    print(best_trial)
