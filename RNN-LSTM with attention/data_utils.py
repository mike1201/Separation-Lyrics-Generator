# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import pickle
import sys
from tensorflow.python.platform import gfile


# 1-1. Define Symbols.

# 1-1-1. Define Special vocabulary symbols.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]
START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = [_UNK]
PAD_ID = 0
UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1
UNK_ID_dict['no_padding'] = 0

# 1-1-2. Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# 1-2. Define Tokenizer
def tokenizer(sentence):
  return sentence.split()  


# 1-3. Make vocabulary list which are in dictionary
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      normalize_digits=True):

  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    
    # 1-3-1. Read data.
    with open(data_path, mode="r", encoding = "utf-8") as f:
      
      # Count the number of line.
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("processing line %d" % counter)
        
        # 1-3-2. Tokenize the line.
        tokens = tokenizer(line)
        
        # 1-3-3. Normalize digits to zero and Make frequency vocab.
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = START_VOCAB_dict['with_padding'] + \
                      sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with open(vocabulary_path, mode="w", encoding = "utf-8") as vocab_file:
        
        # 1-3-4. Save the vocab in "vocabulary_path"
        for w in vocab_list:
          vocab_file.write(w + "\n")

# 1-4. Make index dictionary and word list
def initialize_vocab(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path, mode="r", encoding = "utf-8") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
    # rev_vocab = [word1, word2, ...], vocab = [word1:index1, word2:index2 , ..., ]
  else: # If not exist the file from "3"
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


    
# 1-5. String Sentence --) Index Sentence
def sentence_to_token_ids(sentence, vocabulary, UNK_ID, normalize_digits=True):
    
  # 1-5-1. Tokenize the sentence
  words = tokenizer(sentence)
    
  # 1-5-2. Treat Exception words
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


# 1-6. Tokenize data and turn into toekn_index
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      normalize_digits=True, use_padding=True):

  if not gfile.Exists(target_path):
    
    # 1-6-1. Make index dictionary using "#1-4."
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocab(vocabulary_path)
    
    # 1-6-2. Read raw data.
    with open(data_path, mode="r", encoding = "utf-8") as data_file:
      with open(target_path, mode="w", encoding = "utf-8") as tokens_file:
        
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
            
          # 1-6-3. Decide whether using padding.
          if use_padding:
            UNK_ID = UNK_ID_dict['with_padding'] # 1
          else:
            UNK_ID = UNK_ID_dict['no_padding'] # 0 --) UNK
          
          # 1-6-4. Get idx sentence and write to token_file
          token_ids = sentence_to_token_ids(line, vocab, UNK_ID,normalize_digits)
          # tokenize the data by line
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
        

# 1-7. String data --) token_index / Make word and index dictionary
def prepare_multi_task_data(data_dir, in_vocab_size, out_vocab_size):
    
    # 1-7-1. Get Path of train/dev/test data.
    train_path = data_dir + '/train/train'
    test_path = data_dir + '/test/test'
    
    # 1-7-2. Get the vocab list and indexed vocab(=indexing) in dictionary
    # 1-7-2-1. Make the vocab file path.
    in_vocab_path = os.path.join(data_dir, "in_vocab_%d.txt" % in_vocab_size)
    out_vocab_path = os.path.join(data_dir, "out_vocab_%d.txt" % out_vocab_size)
    
    # 1-7-2-2. Make vocab list and indexed vocab(=indexing) in dictionary
    create_vocabulary(in_vocab_path, 
                      train_path + ".seq.in", 
                      in_vocab_size) # a
    create_vocabulary(out_vocab_path, 
                      train_path + ".seq.out", 
                      out_vocab_size) # b
    
    
    # 1-7-3. String data --) Token_index
    # 1-7-3-1. Get the path of indexing train data set.
    in_seq_train_ids_path = train_path + (".ids%d.seq.in" % in_vocab_size) # a
    out_seq_train_ids_path = train_path + (".ids%d.seq.out" % out_vocab_size) # b
    
    # 1-7-3-2. String data --) Token_idx
    # for train data
    data_to_token_ids(train_path + ".seq.in", 
                      in_seq_train_ids_path, 
                      in_vocab_path)
    data_to_token_ids(train_path + ".seq.out", 
                      out_seq_train_ids_path, 
                      out_vocab_path)
   
    # For test data
    in_seq_test_ids_path = test_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_test_ids_path = test_path + (".ids%d.seq.out" % out_vocab_size)
    
    data_to_token_ids(test_path + ".seq.in", 
                      in_seq_test_ids_path, 
                      in_vocab_path)
    data_to_token_ids(test_path + ".seq.out", 
                      out_seq_test_ids_path, 
                      out_vocab_path)
    
    return [(in_seq_train_ids_path,out_seq_train_ids_path),
            (in_seq_test_ids_path, out_seq_test_ids_path),
            (in_vocab_path, out_vocab_path)]

