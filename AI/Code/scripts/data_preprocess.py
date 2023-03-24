import copy
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

def allience_array(file_path,seqlen,steps):
    alliance_dfs = []
    df = pd.read_csv(file_path, low_memory=False)
    #df = df[:200]
    a = 0

    while a < len(df['alliance_id']-1):
        newRow = []
        allience_id = df['alliance_id'].values[a]

        while allience_id == df['alliance_id'].values[a]:
            row = df.values[a]
            if len(newRow) == 0:
                newRow = row
                for i in range(len(row)):
                    hold = []
                    hold.append(newRow[i])
                    newRow[i] = hold
            elif (len(newRow[0]) == seqlen):
                alliance_dfs.append(copy.deepcopy(newRow))
                for i in range(len(newRow)):
                    newRow[i] = newRow[i][steps:]
                    newRow[i].append(row[i])
            else:
                for i in range(len(row)):
                    newRow[i].append(row[i])
            a = a + 1
        alliance_dfs.append(newRow)

    return alliance_dfs, df


def split_csv_file_by_alliance_id(df, seqlen, steps):

    alliance_dfs = []
    print(len(df))
    count=0
    for alliance_id in df['alliance_id'].unique():

        rows = [x for x in df.values if x[1]==alliance_id]
        count+=1
        '''
        if count%100==0:
            print(count)
        '''
        newRow = []
        for row in rows:
            if row[1] == alliance_id:
                if len(newRow) == 0:
                    newRow = row
                    for i in range(len(row)):
                        hold = []
                        hold.append(newRow[i])
                        newRow[i] = hold
                elif (len(newRow[0]) == seqlen):
                    alliance_dfs.append(copy.deepcopy(newRow))
                    for i in range(len(newRow)):
                        newRow[i] = newRow[i][steps:]
                        newRow[i].append(row[i])
                else:
                    for i in range(len(row)):
                        newRow[i].append(row[i])

        alliance_dfs.append(newRow)

    return alliance_dfs, df


def save_as_df(allienceSet, odf):
    fields = odf.columns
    with open('data.csv', "a", encoding="utf-8") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(allienceSet)


def array_maker(file_path):
    df = pd.read_csv(file_path, low_memory=False)#[:10000]
    array = ["",]
    i = 0
    c = 5000
    a = 0

    while i < len(df.values):

        while c + 1 < (len(df.values)) and (df["alliance_id"].values[c] == df["alliance_id"].values[c + 1]):
            c += 1

        if i == 0:
            array[a] = df[i:c]
        else:
            array.append(df[i:c])

        i = c
        print(i / len(df.values))
        c += 5000
        if c>len(df.values):
            array.append(df[i:len(df.values)])
        a += 1

    return array


file = os.path.join(Path(os.path.split(os.path.dirname(os.path.abspath(os.getcwd())))[0]), "Data/combinedout.csv")
allienceSet = array_maker(file)
print('sets created!')
for i in allienceSet:
    print(i/len(allienceSet))
    all,odf = split_csv_file_by_alliance_id(i,20,1)
    save_as_df(all,odf)