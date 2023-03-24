import numpy as np
import torch
import re
import pandas as pd
import ast
import sys
from scipy.spatial import distance
import torch.nn as nn

def fix_data(data):

    # transfer back to proper list input types
    data = np.array(data)

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

def load_full_data(seq_length):

    train_x = torch.load('Code/train_emb_'+str(seq_length)+'.pt')
    val_x = torch.load('Code/val_emb_'+str(seq_length)+'.pt')
    test_x = torch.load('Code/test_emb_'+str(seq_length)+'.pt')

    x = torch.cat((train_x, val_x),axis=0)
    x = torch.cat((x, test_x),axis=0)

    return x

def show_analysis(top_k_picks, data):
    #data = fix_data(data)
    indexes = [8, 11, 12, 13, 14, 15, 16, 19, 24, 25]
    for i in range(top_k_picks.indices.shape[0]):
        #print('Cluster ' + str(i))
        cluster_risk = []
        #for j in range(top_k_picks.indices.shape[0]):
            #print('Pick ' + str(j))
        idx = top_k_picks.indices[i]
        scores = []
        for k in indexes:
            #risks = ast.literal_eval(str(data[idx, k]))
            #risks = list(map(float, risks))
            risks_avg = np.mean(data[idx,k])
            scores.append(risks_avg)
        cluster_risk.append(scores)

    print('Average risk for top k picks:')
    print(np.mean(np.array(cluster_risk),axis=0))

def filter_chats(data, k, seq_length):
    # Read the CSV file into a DataFrame
    #df = pd.read_csv(csv_path)
    #data = fix_data(df)
    x = load_full_data(seq_length)
    means = []
    for i in data[:,8]:
        means.append(np.median(i))
    top_k_picks = torch.topk(torch.tensor(means), k=k, dim=0, largest=True, sorted=True)
    show_analysis(top_k_picks, data)
    print(top_k_picks.indices.shape)
    emb = []
    for idx in range(0, len(top_k_picks.indices)):
        if idx == 0:
            emb = x[top_k_picks.indices[idx]].reshape(1,x.shape[1])
        else:
            emb = torch.cat((emb, x[top_k_picks.indices[idx]].reshape(1,x.shape[1])),axis=0)

    final_emb = torch.mean(emb,axis=0)

    data = list(data)
    for i in sorted(top_k_picks.indices, reverse=True):
        del data[i]
    data = np.array(data)

    scores = []
    for i in range(0, len(data)):
        data_scores = []
        #for j in range(cluster_centers.shape[0]):
        dist = distance.euclidean(x[i], final_emb)
        #data_scores.append(dist)
        #scores.append(data_scores)
        scores.append(dist)

    # find closest matches
    top_k_picks = torch.topk(torch.tensor(scores),k=10, dim=0, largest=False,sorted=True)
    show_analysis(top_k_picks, data)

