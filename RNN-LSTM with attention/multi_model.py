# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import seq_labeling
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn


class MultiTaskModel(object):
  def __init__(self, 
               source_vocab_size, 
               tag_vocab_size, 
               buckets, 
               word_embedding_size, 
               size, 
               num_layers, 
               max_gradient_norm, 
               batch_size, 
               dropout_keep_prob=1.0, 
               use_lstm=False, 
               bidirectional_rnn=True,
               num_samples=1024, 
               use_attention=False,
               forward_only=False):
    self.source_vocab_size = source_vocab_size
    self.tag_vocab_size = tag_vocab_size
    self.word_embedding_size = word_embedding_size
    self.cell_size = size
    self.num_layers = num_layers
    self.buckets = buckets
    self.batch_size = batch_size
    self.bidirectional_rnn = bidirectional_rnn
    self.global_step = tf.Variable(0, trainable=False)
    
    # If we use sampled softmax, we need an output projection.
    softmax_loss_function = None

    # 2-1. Make multi-layer cells
    def create_cell():
        
      # Add dropout
      if not forward_only and dropout_keep_prob < 1.0:
        single_cell = lambda: BasicLSTMCell(self.cell_size)
        cell = MultiRNNCell([single_cell() for _ in range(self.num_layers)])
        cell = DropoutWrapper(cell,
                                input_keep_prob=dropout_keep_prob,
                                output_keep_prob=dropout_keep_prob)      
      # Not Dropout
      else:
        single_cell = lambda: BasicLSTMCell(self.cell_size)
        cell = MultiRNNCell([single_cell() for _ in range(self.num_layers)])
      return cell
  
    # 2-1-1. Create Forward/Backward cell of encoder.
    self.cell_fw = create_cell()
    self.cell_bw = create_cell()

    # 2-2. Define Placeholder(=input)
    self.encoder_inputs = []
    self.tags = []    
    self.tag_weights = []    
    self.sequence_length = tf.placeholder(tf.int32, [None], 
                                          name="sequence_length")
    
    # 2-2-1. Define Sentence placeholder( =encoder_inputs)
    for i in xrange(buckets[-1][0]): # bucket[-1][0] = encoder_length, xrange --) range
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    
    # 2-2-2. Define tags and tags weights
    for i in xrange(buckets[-1][1]):
      self.tags.append(tf.placeholder(tf.float32, shape=[None], name="tag{0}".format(i)))
      self.tag_weights.append(tf.placeholder(tf.float32, shape=[None],
                                             name="weight{0}".format(i)))
        
        
    # 2-3-5. Get the bi-directional outputs
    base_rnn_output = self.generate_rnn_output()
    encoder_outputs, encoder_state, attention_states = base_rnn_output
    
    
    # 2-4-1. Sequence labeling 
    seq_labeling_outputs = seq_labeling.generate_sequence_output(
                                   self.source_vocab_size,
                                   encoder_outputs, 
                                   encoder_state, 
                                   self.tags, 
                                   self.sequence_length, 
                                   self.tag_vocab_size, 
                                   self.tag_weights,
                                   buckets, 
                                   softmax_loss_function=softmax_loss_function, 
                                   use_attention=use_attention)
    self.tagging_output, self.tagging_loss = seq_labeling_outputs
    
    # 2-4-2. Define Loss.
    self.loss = self.tagging_loss
    
    # 2-5. Define Gradients and SGD and train the model.
    params = tf.trainable_variables()
    if not forward_only:
        
      # 2-5-1. Define Optimizer and gradients
      opt = tf.train.AdamOptimizer()
      gradients = tf.gradients(self.tagging_loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
      self.gradient_norm = norm
        
      # 2-5-2. Train
      self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables())
    
    
  # 2-3. Generate bi-directional RNN-LSTM Output
  def generate_rnn_output(self):
    
    with tf.variable_scope("generate_seq_output"):
      if self.bidirectional_rnn:
        
        # 2-3-1. Embedding Matrix
        embedding = tf.get_variable("embedding",
                                    [self.source_vocab_size,
                                     self.word_embedding_size])
        
        # 2-3-1-1. encoder_inputs --) encoder_emb_inputs
        encoder_emb_inputs = list()
        encoder_emb_inputs = [tf.nn.embedding_lookup(embedding, encoder_input)\
                                for encoder_input in self.encoder_inputs]
        
        # 2-3-2. Get the encoder output.
        rnn_outputs = static_bidirectional_rnn(self.cell_fw,
                                               self.cell_bw, 
                                               encoder_emb_inputs, 
                                               sequence_length=self.sequence_length,
                                               dtype=tf.float32)
        
        # 2-3-2-1. 
        encoder_outputs, encoder_state_fw, encoder_state_bw = rnn_outputs
        
         # 2-3-3. Use final cell state and concatenate
        state_fw = encoder_state_fw[-1]
        state_bw = encoder_state_bw[-1]
        # 2-3-3-1. Get cell state of last data
        encoder_state = tf.concat([tf.concat(state_fw, 1),tf.concat(state_bw, 1)], 1)
        
        # 2-3-4. Reshape encoder_outputs to put in attention_states
        top_states = [tf.reshape(e, [-1, 1, self.cell_fw.output_size \
                                  + self.cell_bw.output_size]) for e in encoder_outputs]
                                    # output_size = embedding_size
        attention_states = tf.concat(top_states, 1)  # [-1,2,2 X emb_dim]
        
        
      # Similar with the above
      else: # Not using bi-directional RNN
            
        embedding = tf.get_variable("embedding", 
                                    [self.source_vocab_size,
                                     self.word_embedding_size]) 
        encoder_emb_inputs = list()
        encoder_emb_inputs = [tf.nn.embedding_lookup(embedding, encoder_input)\
                              for encoder_input in self.encoder_inputs] 
        rnn_outputs = static_rnn(self.cell_fw,
                                 encoder_emb_inputs,
                                 sequence_length=self.sequence_length,
                                 dtype=tf.float32)
        encoder_outputs, encoder_state = rnn_outputs
        # with state_is_tuple = True, if num_layers > 1, 
        # here we use the state from last layer as the encoder state
        state = encoder_state[-1]
        encoder_state = tf.concat(state, 1)
        top_states = [tf.reshape(e, [-1, 1, self.cell_fw.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(top_states, 1)
      return encoder_outputs, encoder_state, attention_states # Input of attention.
  

  def tagging_step(self, session, encoder_inputs, tags, tag_weights, 
                   batch_sequence_length, bucket_id, forward_only):
    
    
    # Check if the sizes match.
    encoder_size, tag_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(tags) != tag_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(tags), tag_size))

    
    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.sequence_length.name] = batch_sequence_length
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      input_feed[self.tags[l].name] = tags[l]
      input_feed[self.tag_weights[l].name] = tag_weights[l]

    
    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.update,
                     self.gradient_norm,
                     self.loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])
    else:
      output_feed = [self.loss]
      for i in range(tag_size):
        output_feed.append(self.tagging_output[i])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3:3+tag_size]
    else:
      return None, outputs[0], outputs[1:1+tag_size]



  def get_batch(self, data, bucket_id):
    
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    batch_sequence_length_list= list()
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input= random.choice(data[bucket_id])
      batch_sequence_length_list.append(len(encoder_input))
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(encoder_input + encoder_pad))
      decoder_pad_size = decoder_size - len(decoder_input)
      decoder_inputs.append(decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
    batch_encoder_inputs = []
    batch_decoder_inputs = []
    batch_weights = []
  
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))                 
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
    return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, 
            batch_sequence_length)



  def get_one(self, data, bucket_id, sample_id):

    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    batch_sequence_length_list= list()
    encoder_input, decoder_input= data[bucket_id][sample_id]
    batch_sequence_length_list.append(len(encoder_input))
    encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
    encoder_inputs.append(list(encoder_input + encoder_pad))
    decoder_pad_size = decoder_size - len(decoder_input)
    decoder_inputs.append(decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
    batch_encoder_inputs = []
    batch_decoder_inputs = []
    batch_weights = []
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(1)], dtype=np.int32))
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(1)], dtype=np.int32))
      batch_weight = np.ones(1, dtype=np.float32)
      for batch_idx in xrange(1):
        if decoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    
    batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
    return (batch_encoder_inputs, batch_decoder_inputs,
            batch_weights, batch_sequence_length)
