from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import sys
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import data_utils
import multi_model
import subprocess
import stat

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 100,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 40000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 40000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "data_lyrics", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps",30000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 6,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")  
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
FLAGS = tf.app.flags.FLAGS


# Check Variables
if FLAGS.max_sequence_length == 0:
    print ('Please indicate max sequence length. Exit')
    exit()
_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]


# 4-1. Read Source / Target data
def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]

  # 4-1-1. Read one line of Source / Target data.
  with open(source_path, mode="r", encoding = "utf-8") as source_file:
    with open(target_path, mode="r", encoding = "utf-8") as target_file:
        source = source_file.readline()
        target = target_file.readline()
        
        counter = 0
        
        # 4-1-2. Set maximum number of lines
        while source and target and (not max_size or counter < max_size):
            
          # Count the line
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
            
          # 4-1-3. Get the length of line
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          
          # 4-1-4. Choose bucket and add one data to corresponding dataset
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids])
              break
                
          # 4-1-5. 
          source = source_file.readline()
          target = target_file.readline()
          
  return data_set

# 4-2. Train/test model
def create_model(session, 
                 source_vocab_size, 
                 target_vocab_size):
    
  # 4-2-1. Create train model.
  with tf.variable_scope("model", reuse=None):
    model_train = multi_model.MultiTaskModel(
          source_vocab_size, 
          target_vocab_size, 
          _buckets,
          FLAGS.word_embedding_size, 
          FLAGS.size, FLAGS.num_layers, 
          FLAGS.max_gradient_norm, 
          FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, 
          use_lstm=True,
          forward_only=False, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          )
    
  # 4-2-2. Create test model.
  with tf.variable_scope("model", reuse=True):
    model_test = multi_model.MultiTaskModel(
          source_vocab_size, 
          target_vocab_size, 
          _buckets,
          FLAGS.word_embedding_size, 
          FLAGS.size, 
          FLAGS.num_layers, 
          FLAGS.max_gradient_norm, 
          FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, 
          use_lstm=True,
          forward_only=True, 
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          )
    
  
  # 4-2-3. Get model paramters or Initialize the model parameters.
  # 4-2-3-1. Get model parameters.
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt:
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model_train.saver.restore(session, ckpt.model_checkpoint_path)
      
  # 4-2-3-2. Initialize the model parameters.    
  else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
  return model_train, model_test


# 4-3.        
def train():
    
  # See parameters.
  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].items():
    print ('%s: %s' % (k, str(v)))
    
  # 4-3-1. Prepare indexing data and correspondiing labels.  
  print("Preparing data in %s" % FLAGS.data_dir)
  vocab_path = ''
  tag_vocab_path = ''
    
  # 4-3-1-1. String data --) token index / Make word and label dictionary.
  date_set = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)
    
  # 4-3-1-2. Get path of each result.
  in_seq_train, out_seq_train = date_set[0]
  in_seq_test, out_seq_test = date_set[1]
  vocab_path, tag_vocab_path = date_set[2]
 
  # Where do we save the result?  
  result_dir = FLAGS.train_dir + '/test_results'
  if not os.path.isdir(result_dir):
      os.makedirs(result_dir)
  current_taging_valid_out_file = result_dir + '/tagging.valid.hyp.txt'
  current_taging_test_out_file = result_dir + '/tagging.test.hyp.txt'
    
  # 4-3-2. Get index dictionary and word list.   
  vocab, rev_vocab = data_utils.initialize_vocab(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocab(tag_vocab_path)
  tag_vocab_inv = dict()
    
    
  for string, i in tag_vocab.items():
        tag_vocab_inv[i] = string
  config = tf.ConfigProto(
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.23),
      #device_count = {'gpu': 2}
  )
    
  with tf.Session(config=config) as sess:
    print("Max sequence length: %d." % _buckets[0][0])
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    
    # 4-3-3. Make train/test model.
    model, model_test = create_model(sess, 
                                     len(vocab), 
                                     len(tag_vocab)
                                     )
    print ("Creating model with " + 
           "source_vocab_size=%d, target_vocab_size=%d" \
           % (len(vocab), len(tag_vocab)))

    # Read data into buckets and compute their sizes.
    print ("Reading train/valid/test data (training set limit: %d)."
           % FLAGS.max_train_data_size)
    
    # 4-3-4. Load data using "# 4-1."
    test_set = read_data(in_seq_test, out_seq_test)
    train_set = read_data(in_seq_train, out_seq_train)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    
    # 4-3-5. Train Loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    best_valid_score = 0
    best_test_score = 0

    while model.global_step.eval() < FLAGS.max_training_steps:
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])
      start_time = time.time()
        
      # 4-3-5-1. get batch
      batch_data = model.get_batch(train_set, bucket_id)
      encoder_inputs,tags,tag_weights,batch_sequence_length = batch_data
      
      step_outputs = model.tagging_step(sess, 
                                        encoder_inputs,
                                        tags,
                                        tag_weights,
                                        batch_sequence_length, 
                                        bucket_id, 
                                        False)
      _, step_loss, tagging_logits = step_outputs
        
        
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d step-time %.2f. Training perplexity %.2f" 
            % (model.global_step.eval(), step_time, perplexity))
        sys.stdout.flush()
        
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        
           
    # Test
    count = 0
    word_list = list()
    ref_tag_list = list()
    hyp_tag_list = list()
    for bucket_id in xrange(len(_buckets)):
        for i in xrange(len(test_set[bucket_id])):
            count += 1
            sample = model_test.get_one(test_set, bucket_id, i)
            encoder_inputs, tags, tag_weights, sequence_length= sample
            step_outputs = model_test.tagging_step(sess,
                                                   encoder_inputs,
                                                   tags,
                                                   tag_weights,
                                                   sequence_length,
                                                   bucket_id,
                                                   True)
            _, step_loss, tagging_logits = step_outputs
            
            lst = []
            string = ""
            for num in encoder_inputs:
                num = num[0]
                word = rev_vocab[num]
                if word == "_PAD" or word == "_UNK":
                    continue
                else:
                    lst.append(word)
                    string = string + word + " "
            string = string + " : "
            string2 = string
            
            for word in tagging_logits:
                word = word[0]
                sort_num = np.argsort(word)
                b = sort_num[39999]
                word = rev_tag_vocab[b]
                if word == "_PAD" or word == "_UNK":
                    continue
                else:
                    lst.append(word)
                    string = string + word + " "
            print(string)
            
            for word in tagging_logits:
                word = word[0]
                sort_num = np.argsort(word)
                b = sort_num[39998]
                word = rev_tag_vocab[b]
                if word == "_PAD" or word == "_UNK":
                    continue
                else:
                    lst.append(word)
                    string2 = string2 + word + " "
            print(string2)
            print("\n")
            
       

def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()
