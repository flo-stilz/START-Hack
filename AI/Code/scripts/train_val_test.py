import numpy as np
import torch
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import csv
import ast
import re
import pytorch_lightning as pl
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, BertTokenizer
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from models.regressor_models import Bert

def fix_data(data):

    # transfer back to proper list input types
    data = np.array(data)#[:1000]

    remove_index = []
    for i in range(0,len(data)):
        for j in range(len(data[i])):
            #print(data[i,j])
            #print(i)
            try:
                data[i,j] = ast.literal_eval(str(data[i,j]))
                if j in [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
                    data[i,j] = list(map(float, data[i,j]))
            except ValueError:
                remove_index.append(i)
                continue
    # remove nan values:
    data = list(data)
    for idx in sorted(remove_index, reverse=True):
        del data[idx]
    data = np.array(data)

    # combine sentences:
    texts = []
    for i in range(0, len(data)):
        for j in range(len(data[i,4])):
            if j==0:
                text = data[i,4][j]
            else:
                text += ' '+data[i,4][j]

        text = deEmojify(text)
        texts.append(text)
    data = np.concatenate((data, np.array(texts).reshape(len(data), 1)),axis=1)

    return data

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def get_emb(dataset_path, opt):

    # load data
    data = pd.read_csv(dataset_path)

    data_np = fix_data(data)
    print('cleaned data!')
    #data_inp = get_input_samples(data, opt)

    data_tok = tokenize(data_np, opt)

    print('tokenized data!')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    #model = globals()[opt.model](opt=opt).double()
    model = Bert(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    print(device)
    model = model.to(device)

    emb = []
    flag = False
    for i in range(0, int((len(data_tok['input_ids']))/opt.batch_size)+1):
        print((i*opt.batch_size)/len(data_tok['input_ids']))
        if ((i+1)*opt.batch_size)>=len(data_tok['input_ids']):
            end = len(data_tok['input_ids'])
            flag = True
        else:
            end = ((i+1)*opt.batch_size)
        #inp = {key + '_t': torch.tensor(val[(i * opt.batch_size):((i + 1) * opt.batch_size)]) for key, val in data_tok.items()}
        input_ids = torch.tensor(data_tok['input_ids'][(i*opt.batch_size):end]).to(device)
        attention_mask = torch.tensor(data_tok['attention_mask'][(i * opt.batch_size):end]).to(device)
        sub_emb = model(input_ids, attention_mask)
        if i==0:
         emb = sub_emb.detach().cpu()
        else:
         emb = torch.cat((emb, sub_emb.detach().cpu()),axis=0)

        if flag:
            break
        # free mem:
        input_ids.detach().cpu()
        attention_mask.detach().cpu()
        del input_ids
        del attention_mask
        torch.cuda.empty_cache()
        #print(torch.cuda.memory_allocated())
        #print(torch.cuda.memory_reserved())

    train_emb, val_emb, test_emb, data_train, data_val, data_test = split_sets(emb, data_np)


    # store embedding files:
    torch.save(train_emb, 'train_emb_'+str(opt.seq_length)+'.pt')
    torch.save(val_emb, 'val_emb_'+str(opt.seq_length)+'.pt')
    torch.save(test_emb, 'test_emb_'+str(opt.seq_length)+'.pt')

    # store data in sets:
    fields = data.columns.values.tolist() + ['chat']
    with open('train_data.csv', 'w', encoding="utf-8") as f:

        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(data_train)

    with open('val_data.csv', 'w', encoding="utf-8") as f:

        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(data_val)

    with open('test_data.csv', 'w', encoding="utf-8") as f:

        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(data_test)


def tokenize(data, opt):

    data_pre_tok = pre_tokenize_text(data)
    tokenizer = AutoTokenizer.from_pretrained(opt.model, add_prefix_space=True)
    encodings = tokenizer(list(data_pre_tok), is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=512)
    encodings.pop("offset_mapping")

    return encodings


def pre_tokenize_text(data):
    # nlp = spacy.blank('en')
    data = np.array(data)
    data_pre_tok = []
    for i in range(0, len(data)):
        doc = word_tokenize(data[i, -1]) # find correct index
        data_pre_tok.append(doc)

    return data_pre_tok

def split_sets(emb, data):

    train_emb, testy_emb, data_train, data_testy = train_test_split(emb, data, test_size=0.2,random_state=1)  # set to (0,7,16) for images
    val_emb, test_emb, data_val, data_test = train_test_split(testy_emb, data_testy, test_size=0.5, random_state=1)

    return train_emb, val_emb, test_emb, data_train, data_val, data_test

'''
def get_input_samples(data, opt):

    return data_inp
'''

def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Prediction Task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, required=False, default=8,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=1e-3,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
                        help='Dropout of model')
    parser.add_argument('--reg', type=float, required=False, default=0,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=1.00,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', type=int, required=False, default=1,
                        help='Set to 1 if no gradient accumulation is desired else enter desired value for accumulated batches before gradient step')
    parser.add_argument('--epochs', type=int, required=False, default=100,
                        help='Set maxs number of epochs.')
    parser.add_argument('--gpu', type=int, required=False, default=0,
                        help='Set the name of the GPU in the system')
    parser.add_argument('--model', type=str, required=False, default="distilbert-base-uncased",
                        help='Model architecture')
    parser.add_argument('--rel_pred', action='store_true',
                        help='Add reliability prediction head')
    parser.add_argument('--num_layers', type=int, required=False, default=1,
                        help='Number of Linear Layers in regressor')
    parser.add_argument('--hidden_size', type=int, required=False, default=50,
                        help='Number of hidden units per layer in regressor')
    parser.add_argument('--hp_tuning', action='store_true',
                        help='Perform Random Search to tune hps')
    parser.add_argument('--trials', type=int, required=False, default=200,
                        help='Number of trials for hp tuning')
    parser.add_argument('--show_scores', type=bool, required=False, default=True,
                        help='Log ci and r2m scores')
    parser.add_argument('--standardize', action='store_true',
                        help='standardize data')
    parser.add_argument('--no_ckpt', action='store_true',
                        help='Do not save model ckpt')
    parser.add_argument('--use_linear_skip', action='store_true',
                        help='Use Skip Connection Layers')
    parser.add_argument('--adj_lr', action='store_true',
                        help='Adjust lr by dividing it by 10 after val-loss plateaued')
    parser.add_argument('--num_models', type=int, required=False, default=2,
                        help='Amount of models used in ensemble method')
    parser.add_argument('--freeze', type=int, required=False, default=0,
                        help='Set how many layers to freeze')
    parser.add_argument('--freeze_emb', action='store_true',
                        help='Freeze embedding layer')
    parser.add_argument('--all-emb', action='store_true',
                        help='Use all embeddings not just CLS')
    parser.add_argument('--seq_length', type=int, required=True,
                        help='Amount of chat messages included')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = init_parser()
    dataset_path = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/data.csv")

    get_emb(dataset_path, opt)