# Load Data script

import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, BertTokenizer
from nltk.tokenize import word_tokenize

class Dataset:
    def __init__(self, mode, X, opt, inference=False):
        self.mode = mode
        self.inference = inference
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if self.inference and self.mode=='test':
            self.X, self.p_names = torch.tensor(np.array(X[0],dtype=float)), X[1]
            self.y = torch.tensor(np.array(y, dtype=float))
        else:
            self.X = np.array(X[:,-1])
            means = []
            for i in range(len(X)):
                means.append((np.mean(X[i,12])+1)/8)
            #print(means) # only for debugging
            self.y = torch.tensor(np.array(means,dtype=float))
        self.opt = opt

        data_pre_tok = self.pre_tokenize_text(self.X)
        tokenizer = AutoTokenizer.from_pretrained(opt.bert_model, add_prefix_space=True)
        self.encodings = tokenizer(list(data_pre_tok), is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=512)
        self.encodings.pop("offset_mapping")
        
        if not self.mode.lower() in ["train", "val", "test"]:
            raise AssertionError

    def pre_tokenize_text(self, data):
        data = np.array(data)
        data_pre_tok = []
        for i in range(0, len(data)):
            doc = word_tokenize(data[i])  # find correct index
            data_pre_tok.append(doc)
        return data_pre_tok

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.y[idx]
        
        return item
    
if __name__ == '__main__':
    
    dataset_path = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Seasons_Stats.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_split(pd.read_csv(dataset_path))

    val_data = Dataset(mode="val", X=X_val, y=y_val, opt=None)
    
