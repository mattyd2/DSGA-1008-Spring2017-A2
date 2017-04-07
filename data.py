import os
import torch
import nltk
import numpy as np
from sklearn.model_selection import train_test_split


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, brown):
        self.dictionary = Dictionary()
        if brown:
            self.data = self.build_data()
            self.train = self.tokenize_brown(self.data[0])
            self.valid = self.tokenize_brown(self.data[1])
            self.test = self.tokenize_brown(self.data[2])
        else:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def build_data(self):
        brown = nltk.corpus.brown.sents(categories=[
            'adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
            'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance',
            'science_fiction'])
        x = np.array(brown)
        x_train, test_val, _, __ = train_test_split(x, x, test_size=0.3, random_state=42)
        x_val, x_test, _, __ = train_test_split(test_val, test_val, test_size=0.2, random_state=42)
        data = [x_train, x_val, x_test]
        return data

    def tokenize(self, path):
        print("Tokenizing Data...")
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        print("Vectorizing Data...")
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def tokenize_brown(self, data):
        print("Tokenizing Data...")
        tokens = 0
        for line in data:
            words = line + ['<eos>']
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        print("Vectorizing Data...")
        ids = torch.LongTensor(tokens)
        token = 0
        for line in data:
            words = line + ['<eos>']
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1
        return ids
