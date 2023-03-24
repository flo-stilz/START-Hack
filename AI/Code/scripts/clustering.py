import pandas as pd
import torch
import numpy as np
import argparse
import re
import os
import ast
import csv
from datetime import datetime
from pathlib import Path
from kmeans_pytorch import kmeans
from scipy.spatial import distance

# data
def clustering(num_clusters, train_x):

    '''
    x = np.random.randn(data_size, dims) / 6
    x = torch.from_numpy(x)
    '''

    # TODO: try different cluster alg:
    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=train_x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )

    return cluster_ids_x, cluster_centers

def load_data(opt):

    train_x = torch.load('train_emb_'+str(opt.seq_length)+'.pt')

    return train_x

def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Prediction Task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_clusters', type=int, required=True,
                        help='Number of clusters!')
    parser.add_argument('--gpu', type=int, required=True,
                        help='GPU Number')
    parser.add_argument('--dims', type=int, required=False, default=768,
                        help='Embedding dim!')
    parser.add_argument('--seq_length', type=int, required=True,
                        help='Seq Length!')
    parser.add_argument('--k', type=int, required=False, default=5,
                        help='Top k picks to look at per cluster')
    parser.add_argument('--all-emb', action='store_true',
                        help='Use all embeddings not just CLS')

    opt = parser.parse_args()

    return opt

def load_full_data(opt):

    train_x = torch.load('train_emb_'+str(opt.seq_length)+'.pt')
    val_x = torch.load('val_emb_'+str(opt.seq_length)+'.pt')
    test_x = torch.load('test_emb_'+str(opt.seq_length)+'.pt')

    x = torch.cat((train_x, val_x),axis=0)
    x = torch.cat((x, test_x),axis=0)

    return x


def top_k_matches(cluster_centers, k, emb, data):

    #cos = nn.CosineSimilarity(dim=0)
    scores = []
    for i in range(0, len(emb)):
        data_scores = []
        for j in range(cluster_centers.shape[0]):
            dist = distance.euclidean(emb[i], cluster_centers[j])
            data_scores.append(dist)
        scores.append(data_scores)

    # find closest matches
    top_pick = torch.min(torch.tensor(scores),dim=0)
    print(top_pick)
    top_k_picks = torch.topk(torch.tensor(scores),k=k, dim=0, largest=False,sorted=True)
    print(top_k_picks)

    show_analysis(top_k_picks, data)
    '''
    count=0
    for indices in top_k_picks.indices:
        count+=1
        print(str(count))
        for idx in indices:
            print(data[idx])
    '''

def show_analysis(top_k_picks, data):
    #data = fix_data(data)
    indexes = [8, 11, 12, 13, 14, 15, 16, 19, 24, 25]
    for i in range(top_k_picks.indices.shape[1]):
        print('Cluster ' + str(i))
        cluster_risk = []
        for j in range(top_k_picks.indices.shape[0]):
            #print('Pick ' + str(j))
            idx = top_k_picks.indices[j,i]
            #print(data[idx,-1])
            sub_risks_avg = []
            for k in indexes:
                risks = ast.literal_eval(str(data[idx, k]))
                risks = list(map(float, risks))
                risks_avg = np.mean(risks)
                sub_risks_avg.append(risks_avg)
            cluster_risk.append(sub_risks_avg)

        print('Average Cluster risk:')
        print(np.mean(np.array(cluster_risk),axis=0))


def fix_data(data):

    # transfer back to proper list input types
    fields = data.columns.values.tolist()+['chat']
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

    with open('clean_data.csv', 'w', encoding="utf-8") as f:

        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(data)

    return data

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

if __name__ == '__main__':

    # init console arg:
    opt = init_parser()

    # load train data
    train_x = load_data(opt)

    # run clustering alg:
    cluster_ids_x, cluster_centers = clustering(opt.num_clusters, train_x)
    print(cluster_centers)
    print(cluster_ids_x.shape)
    print(cluster_centers.shape)
    torch.save(cluster_centers, 'cluster_centers.pt')

    x = load_full_data(opt)
    file = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/clean_data.csv")
    data = pd.read_csv(file)
    top_k_matches(cluster_centers, opt.k, x, np.array(data))
    # TODO: store cluster_ids_x and cluster centers