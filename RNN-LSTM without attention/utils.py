# -*- coding: utf-8 -*-
import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools
import time
import sys

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, nwords, encoding=None):
        # 1-1
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        # 1-2
        input_file = os.path.join(data_dir, "input.txt") # os.path.join : get whole route
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        print("reading text file")
        self.preprocess_nwords(input_file, vocab_file, tensor_file, encoding, nwords)
        '''
        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file, encoding)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        '''

        self.create_batches()
        self.reset_batch_pointer()

    def clean_str(self,string):
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)  
        return string.strip().lower()


    # 1-3 
    # 1-4
    def build_vocab(self, sentences): # 1-4-1
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # 1-4-2
        word_counts = collections.Counter(sentences)# word_counts = { word1:count1, word2:count2,..}
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv)) # vocabulary_inv (= Word Set) = [word1, word2, ...]
     
        # 1-4-3
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # vocabulary (= word_index) = {word1:1, word2:2, ... }
        print(len(vocabulary))
        return [vocabulary, vocabulary_inv]
  # 1-4
    def build_vocab_nwords(self, sentences, n_words): # 1-4-1
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # 1-4-2
        word_counts = collections.Counter(sentences)# word_counts = { word1:count1, word2:count2,..}
        order = word_counts.most_common(n_words-1)
        count = [['UNK', -1]]
        count.extend(order)
        vocabulary_inv = [x[0] for x in order]
        vocabulary_inv = list(sorted(vocabulary_inv)) # vocabulary_inv (= Word Set) = [word1, word2, ...]
      
        # 1-4-3
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # vocabulary (= word_index) = {word1:1, word2:2, ... }
        return [vocabulary, vocabulary_inv, count]
       

    # 1-5
    def preprocess(self, input_file, vocab_file, tensor_file, encoding):
        # 1-5-1
        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        # 1-5-2 (1-3)
        data = self.clean_str(data)
        x_text = data.split() # x_text = [word1, word2, ...?]
        print("text split is over")
        print(x_text[0:10])
        print(len(x_text))
        time.sleep(5)
     

        # 1-5-3 (1-4)
        self.vocab, self.words = self.build_vocab(x_text)
        self.vocab_size = len(self.words)
        print("Building dictionary is over")
        time.sleep(5)
        # self.vocab = {word1 :1, word2:2, ...}, self.words = [word1, word2, ...]
        
        # 1-5-4
        with open(vocab_file, 'wb') as f: # vocab_file = [word1, word2, ...]
            cPickle.dump(self.words, f)

        # 1-5-5
        self.tensor = np.array(list(map(self.vocab.get, x_text))) # self.tensor = [300,25,73 ... ]
        # 1-5-6
        np.save(tensor_file, self.tensor) # Save the data to data.npy

    # 1-5
    def preprocess_nwords(self, input_file, vocab_file, tensor_file, encoding, nwords):
        # 1-5-1
        with codecs.open(input_file, "r", encoding=encoding) as f:
            data = f.read()

        # 1-5-2 (1-3)
        data = self.clean_str(data)
        x_text = data.split() # x_text = [word1, word2, ...?]
        print("text split is over")
        print(x_text[0:10])
        print(len(x_text))
        time.sleep(5)  

        # 1-5-3 (1-4)
        self.vocab, self.words, count = self.build_vocab_nwords(x_text, nwords)
        self.vocab_size = len(self.words)
        print("Building dictionary is over")
        print(len(self.vocab))
        time.sleep(5)

        # self.vocab = {word1 :1, word2:2, ...}, self.words = [word1, word2, ...]
        
        # 1-5-4
        with open(vocab_file, 'wb') as f: # vocab_file = [word1, word2, ...]
            cPickle.dump(self.words, f)

        index_data = []

        # 1-5-5
        for word in x_text:
            if word in self.vocab:
                index = self.vocab[word]
            else:
                index = 0
                count[0][1] +=1
            index_data.append(index)
        self.tensor = np.array(index_data)
        print("indexing on data is done")
        np.save(tensor_file, self.tensor) # Save the data to data.npy


    # 1-6
    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        
    # 1-7
    def create_batches(self):
        
        # 1-7-1.
        self.num_batches = int(self.tensor.size / (self.seq_length * self.batch_size))  # self.tenser.size = Number Of Whole Words
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."
            
        # 1-7-2.
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        
        # 1-7-3.
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        
        # 1-7-4.
        xdata_reshape = xdata.reshape(self.batch_size, -1)
        ydata_reshape = ydata.reshape(self.batch_size, -1)
        # size = [batch_size , num_batches] ( because we cut 'residual data' at above )
        
        self.x_batches = np.split(xdata_reshape, self.num_batches, 1)
        self.y_batches = np.split(ydata_reshape, self.num_batches, 1)
        # # self.x_batches = [batch1, batch2, ...], batch = array( ... )
    
    # 1-8
    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    # 1-9
    def reset_batch_pointer(self):
        self.pointer = 0
