import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np

from beam import BeamSearch

# 2
class Model():
    def __init__(self, args, infer=False): # args is defined in train.py by parser
        # 2-1
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1
                
        # 2-2
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
                       
        # 2-3
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size) # output size = args.rnn_size
            cells.append(cell)            # cells = [cell1, cell2]
        self.cell = cell = rnn.MultiRNNCell(cells)
        
        
        # 2-4 
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        
        
        # 2-5
        self.initial_state = cell.zero_state(args.batch_size, tf.float32) 
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32) # batch_pointer
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1) # tensor : batch --) batch tensor + 1
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False) # epoch_pointer
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)
        
        # For Visualization
        def variable_summaries(var):         
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

                
        # 2-6       
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            
            with tf.device("/gpu:0"):
                # 2-7
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                
                # 2-8
                # [Index1, Index2, ... ] --) [Seq1, Seq2, .. ], Seq = [embed1, embed2, ...]
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # Can`t Understand
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        
        # 2-9
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        
        # 2-10
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        
        # 2-11
        loss = legacy_seq2seq.sequence_loss_by_example(
                logits = [self.logits],
                targets = [tf.reshape(self.targets, [-1])],
                weights = [tf.ones([args.batch_size * args.seq_length])],
                average_across_timesteps = args.vocab_size)
        self.cost = tf.reduce_sum(loss) / ( args.batch_size * args.seq_length )
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        
        # 2-12
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=0, width=4):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime, width):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys()))
            prime_labels = [vocab.get(word, 0) for word in prime.split()]
            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)),
                            prime_labels)
            samples, scores = bs.search(None, None, k=width, maxsample=num)
            return samples[np.argmin(scores)]

        ret = ''
        if pick == 1:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime  = random.choice(list(vocab.keys()))
            print (prime)
            for word in prime.split()[:-1]:
                print (word)
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word,0)
                feed = {self.input_data: x, self.initial_state:state}
                [state] = sess.run([self.final_state], feed)

            ret = prime
            word = prime.split()[-1]
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {self.input_data: x, self.initial_state:state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)

                pred = words[sample]
                ret += ' ' + pred
                word = pred
        elif pick == 2:
            pred = beam_search_pick(prime, width)
            for i, label in enumerate(pred):
                ret += ' ' + words[label] if i > 0 else words[label]
        return ret
