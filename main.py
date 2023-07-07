from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import underthesea
import os
import torch
from underthesea import text_normalize

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy

import matplotlib.pyplot as plt
import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from data import IMDB_dataset
from transformers import BertModel, BertTokenizer
from model import MyModel
from model import get_criterion_and_optimizer
from train import train_model
from train import test_model
from data import word_2_vec

def main ():
    train_dataset, train_label, test_dataset, test_label = IMDB_dataset()
    
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # make data
    print("create train tensor dataset")
    tensor_train_dataset, tensor_masked_train = word_2_vec(train_dataset, tokenizer, 256)
    print("create test tensor dataset")
    tensor_test_dataset, tensor_masked_test = word_2_vec(test_dataset, tokenizer, 256)
    
    batch_size = 32
    dataset_train = TensorDataset(tensor_train_dataset, torch.tensor(train_label), tensor_masked_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = TensorDataset(tensor_test_dataset, torch.tensor(test_label), tensor_masked_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    
    model_bert = BertModel.from_pretrained(model_name)
    
    model_finetuning = MyModel(output_size=2, drop_out=0.1)
    criterion, optimizer = get_criterion_and_optimizer(model=model_finetuning)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # criterion, optimizer
        train_model(model = model_finetuning, dataloader = dataloader_train, criterion = criterion, optimizer = optimizer, epoch = epoch, num_epochs = num_epochs)
        test_model(model = model_finetuning, dataloader = dataloader_test, criterion = criterion)
    
if __name__ == '__main__':
    main()
    