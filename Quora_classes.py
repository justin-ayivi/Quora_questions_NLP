import numpy as np
import pandas as pd
import torch
import gensim



class Quora_dataset(torch.utils.data.Dataset):
    # enable to create a dataset for the data loader
    # https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
    def __init__(self, df1):
        self.df1 = df1
        self.prepare_loader()

    def prepare_loader(self):
        # create to dictionnary for each row of the dataframe
        values = self.df1.values
        cles = self.df1.columns
        questions = [{cles[0] : vals[0],\
            cles[1] : vals[1]} for vals in values]
        self.sample = list(zip(questions, values[:, 2]))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]


class Quora_dataset2(torch.utils.data.Dataset):
    # enable to create a dataset for the data loader
    # https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
    def __init__(self, df1, vocab):
        self.df1 = df1
        self.MAX_SEQUENCE_LENGTH = 30
        self.vocab = vocab
        self.prepare_loader()

    def prepare_loader(self):
        # create to dictionnary for each row of the dataframe
        values = self.df1.values
        cles = self.df1.columns
        questions = []
        for vals in values:
            #converts into a list:
            question1_list = gensim.utils.simple_preprocess(vals[0].encode('utf-8'))
            question1_list = [word for word in question1_list if word in self.vocab][:self.MAX_SEQUENCE_LENGTH]
            question2_list = gensim.utils.simple_preprocess(vals[0].encode('utf-8'))
            question2_list = [word for word in question2_list if word in self.vocab][:self.MAX_SEQUENCE_LENGTH]
            questions.append({cles[0] : question1_list, cles[1] : question2_list})
        self.sample = list(zip(questions, values[:, 2]))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]