# coding: utf8

import os
import sys
import random

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

csv_dir = "memory/"
csv_list = sorted(os.listdir(csv_dir))

table = np.empty((10000, 512))

for i, csv_name in enumerate(csv_list):
    csv_path = csv_dir + csv_name

    with open(csv_path) as f:
        txt = f.read()
    state, _ = txt.split(",next_state,")
    state = state.split(",")
    state = state[:512]
    state = [float(x) for x in state]
    state = np.asarray(state).reshape((1, -1))
    table[i] = state

    if i % 100 == 0:
        print(i)

pd.DataFrame(table).to_csv("raw.csv")

sc = StandardScaler()
table_std = sc.fit_transform(table)

pca = PCA()
pca.fit(table_std)
# データを主成分空間に写像
table_pca = pca.transform(table_std)
print(table_pca)

pd.DataFrame(table_pca[:, :10]).to_csv("pca.csv")
