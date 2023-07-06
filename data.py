import math
import torchtext.datasets
import os
from preprocess import preprocess_text
from preprocess import padding
import nltk
import copy
import torch

from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

from gensim.utils import simple_preprocess
import transformers
import underthesea

def word_2_vec(datasets, tokenizer, max_len):

    tensor_train_dataset = []
    tensor_masked_train = []
    for datapoint in datasets:

        encode = tokenizer.encode_plus(  datapoint,
                                    padding='max_length',
                                    truncation=True,
                                    add_special_tokens=True,
                                    max_length = max_len,
                                    return_tensors='np',
                                    return_token_type_ids=True,
                                    return_attention_mask=True
        )
        encoded_sent = encode['input_ids'].flatten()
        mask =  encode['attention_mask'].flatten()
        tensor_masked_train.append(mask)
        tensor_train_dataset.append(encoded_sent)


    tensor_train_dataset = torch.tensor(tensor_train_dataset)
    tensor_masked_train = torch.tensor(tensor_masked_train)
    return tensor_train_dataset, tensor_masked_train

def make_data(path_folder, label):

    result = []

    for filename in os.listdir(path_folder):
        if filename.endswith(".txt"):

            file_path = path_folder + "/" + filename
            with open(file_path, "r") as file:
                file_contents = file.read()
                file_contents = preprocess_text(file_contents)
                file_contents = "<sos> " + file_contents + " <eos>"
                # file_contents = padding(file_contents, max_len=1024)
                result.append(file_contents)
    return result, [label for _ in range(len(result))]

def IMDB_dataset():
    # http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    # tar -xzf aclImdb_v1.tar.gz
    vocabulary_size = 10000
    path_folder_train_pos = "./aclImdb/train/pos"
    path_folder_train_neg = "./aclImdb/train/neg"

    path_folder_test_pos = "./aclImdb/test/pos"
    path_folder_test_neg = "./aclImdb/test/neg"

    train_sentence_pos, train_label_pos = make_data(path_folder_train_pos, 1)
    train_sentence_neg, train_label_neg = make_data(path_folder_train_neg, 0)

    test_sentence_pos, test_label_pos = make_data(path_folder_test_pos, 1)
    test_sentance_neg, test_label_neg = make_data(path_folder_test_neg, 0)

    train_dataset =  train_sentence_pos + train_sentence_neg
    train_label = train_label_pos + train_label_neg

    test_dataset = test_sentence_pos + test_sentance_neg
    test_label = test_label_pos + test_label_neg

    return train_dataset, train_label, test_dataset, test_label
class DataIMDB():
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.train_dataset, self.train_label, self.test_dataset, self.test_label = self.IMDB_dataset()
        
        print("trainning word2vec")
        # self.modelWord2Vec = self.train_model_word2vec(train_data=self.train_dataset)
        self.model_name = 'bert-base-uncased'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        self.train, self.masked_train = word_2_vector(datasets=self.train_dataset, tokenizer=self.tokenizer)
        self.test, self.masked_test = word_2_vector(datasets=self.test_dataset, tokenizer=self.tokenizer)
    
    def get_data_tensor_and_label(type_data):
        if (type_data == "train"):
            print("get tensor data trainning")
            return self.train, torch.tensor(self.train_label)
        
        if (type_data == "test"):
            print("get tensor data testing")
            return self.test, torch.tensor(self.test_label)
        
    def IMDB_dataset(self):
        # http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
        # tar -xzf aclImdb_v1.tar.gz
        vocabulary_size = 10000
        path_folder_train_pos = "./aclImdb/train/pos"
        path_folder_train_neg = "./aclImdb/train/neg"
        
        path_folder_test_pos = "./aclImdb/test/pos"
        path_folder_test_neg = "./aclImdb/test/neg"
        
        train_sentence_pos, train_label_pos = self.make_data(path_folder_train_pos, 1)
        train_sentence_neg, train_label_neg = self.make_data(path_folder_train_neg, 0)
        
        test_sentence_pos, test_label_pos = self.make_data(path_folder_test_pos, 1)
        test_sentance_neg, test_label_neg = self.make_data(path_folder_test_neg, 0)
        
        train_dataset =  train_sentence_pos + train_sentence_neg
        train_label = train_label_pos + train_label_neg
        
        test_dataset = test_sentence_pos + test_sentance_neg
        test_label = test_label_pos + test_label_neg
        
        return train_dataset, train_label, test_dataset, test_label

    def make_data(self, path_folder, label):
        result = []
        for filename in os.listdir(path_folder):
            if filename.endswith(".txt"):
                file_path = path_folder + "/" + filename
                with open(file_path, "r") as file:
                    file_contents = file.read()
                    file_contents = preprocess_text(file_contents)
                    file_contents = "<sos> " + file_contents + " <eos>"
                    file_contents = padding(file_contents, max_len=512)
                    result.append(file_contents)
        return result, [label for _ in range(len(result))]

    def train_model_word2vec(self, train_data, vector_size=100, window=5):
        data_train_gensim = [sentence.split(' ') for sentence in train_data]
        model = Word2Vec(sentences=data_train_gensim, vector_size=vector_size, window=5, min_count=1, workers=4)
        return model

class DataLoaderIMDB(Dataset):
    def __init__(self, sentence, masked, label):
        # super(DataLoaderIMDB, self).__init__()
        self.sentence = sentence
        self.masked = masked
        self.label = label
    
    def __len__(self):
        return len(label)
    
    def __getitem__(self, index):
        return self.sentence[index], self.masked[index], self.label[index]
