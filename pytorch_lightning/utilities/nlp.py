import nltk
import numpy as np
import torch
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_words, all_words):
    tokenized_words = [stem(w) for w in tokenized_words]
    bag = np.zeros(len(all_words))
    for idx, w in enumerate(all_words):
        if w in tokenized_words:
            bag[idx] = 1.0
    return bag


def create_nlp_data(X, y, matrix_y: bool = False, device: str = "cpu", test_size: float = 0.25):
    data = []
    labels = {}
    labels_r = {}
    idx = 0
    words = []
    for label in y:
        if label not in list(labels.keys()):
            idx += 1
            labels[label] = 1
    for X_batch, y_batch in tqdm(zip(X, y)):
        X_batch = tokenize(X_batch)
        new_X = []
        for Xb in X_batch:
            new_X.append(stem(Xb))
        words.extend(new_X)
        if matrix_y is True:
            data.append([new_X, np.eye(labels[y_batch], len(labels))[labels[y_batch] - 1]])
        else:
            data.append([new_X, labels[y_batch]])
    words = sorted(set(words))
    np.random.shuffle(words)
    np.random.shuffle(data)
    X = []
    y = []
    for sentence, tag in tqdm(data):
        X.append(bag_of_words(sentence, words))
        y.append(tag)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train = torch.from_numpy(np.array(X_train)).to(device).float()
    y_train = torch.from_numpy(np.array(y_train)).to(device).float()
    X_test = torch.from_numpy(np.array(X_test)).to(device).float()
    y_test = torch.from_numpy(np.array(y_test)).to(device).float()
    return X_train, X_test, y_train, y_test, X, y, data, words, labels, labels_r
