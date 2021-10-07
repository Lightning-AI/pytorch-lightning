import os

import cv2
import numpy as np
from tqdm import tqdm


def load_data(data_dir: str, matrix: bool = False, idx_clf: bool = False):
    """
    data_dir : data_dir is the directory that you want to load the data from
    matrix : if you want y as [0,1,0]
    idx_clf : if you want y as 1 or 2 or 3
    """
    data = []
    labels = {}
    labels_r = {}
    idx = 0
    if matrix:
        for label in os.listdir(data_dir):
            idx += 1
            labels[label] = idx
            labels_r[idx] = label
        for folder in tqdm(os.listdir(data_dir)):
            for file in tqdm(os.listdir(f"{data_dir}{folder}/")):
                img = cv2.imread(f"{data_dir}{folder}/{file}")
                img = cv2.resize(img, (56, 56))
                img = img / 255.0
                data.append([img, np.eye(labels[folder] + 1, len(labels))[labels[folder] - 1]])
        X = []
        y = []
        for d in tqdm(data):
            X.append(d[0])
            y.append(d[1])
    if idx_clf:
        for folder in os.listdir(data_dir):
            idx += 1
            for file in os.listdir(f"{data_dir}{folder}/"):
                img = cv2.imread(f"{data_dir}{folder}/{file}")
                img = cv2.resize(img, (56, 56))
                img = img / 255.0
                data.append([img, idx])
        X = []
        y = []
        for d in data:
            X.append(d[0])
            y.append(d[1])
    return X, y, labels, labels_r, idx, data
