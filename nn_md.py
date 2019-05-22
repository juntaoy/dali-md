from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import json
import threading
import numpy as np
import tensorflow as tf

import util
import md_ops

class NNMD(object):
  def __init__(self, config):
    self.config = config

    input_props = self.add_model_specific_valuables(config)
    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"],
                                               staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam": tf.train.AdamOptimizer,
      "sgd": tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def add_model_specific_valuables(self,config):
    raise NotImplementedError

  def tensorize_example(self, example, is_training):
    raise NotImplementedError

  def get_predictions_and_loss(self, inputs):
    raise NotImplementedError

  def get_top_mentions(self, num_words,candidate_starts, candidate_ends,candidate_mention_scores):
    if self.config['mention_selection_method'] == 'high_f1':
      k = num_words #will be filtered later
    else: #default is high_recall
      k = tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"]))

    top_span_indices = md_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                            tf.expand_dims(candidate_starts, 0),
                                            tf.expand_dims(candidate_ends, 0),
                                            tf.expand_dims(k, 0),
                                            True)  # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]

    top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]

    if self.config['mention_selection_method'] == 'high_f1':
      sigmoid_span_mention_scores = tf.nn.sigmoid(top_span_mention_scores)
      threshold_mask = tf.greater_equal(sigmoid_span_mention_scores,self.config['mention_selection_threshold'])
      top_span_starts = tf.boolean_mask(top_span_starts,threshold_mask)
      top_span_ends = tf.boolean_mask(top_span_ends,threshold_mask)


    return top_span_starts,top_span_ends

  def start_enqueue_thread(self, session,cross_validate=-1,nfold=1):
    with open(self.config["train_path"]) as f:
      if cross_validate <0:
        train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
      else:
        train_examples = [json.loads(jsonline) for i,jsonline in enumerate(f.readlines()) if i%nfold != cross_validate]
    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()



  def tensorize_mentions(self, mentions):
    starts,ends = [],[]
    if len(mentions) > 0:
      for m in mentions:
        starts.append(m[0])
        ends.append(m[1])
    return np.array(starts), np.array(ends)

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.reduce_any(same_span,axis=0) #[num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)


  def sigmoid_loss(self, span_scores, span_labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(span_labels),logits=span_scores)
    loss = tf.reduce_sum(loss)
    return loss

  def get_mention_scores(self, span_emb,dropout):
    with tf.variable_scope("mention_scores"):
      return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, dropout) # [k, 1]


  def lstm_contextualize(self, text_emb, text_len, text_len_mask,lstm_dropout,flatten_emb=True):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs
    if flatten_emb:
      return self.flatten_emb_by_sentence(text_outputs, text_len_mask)
    else:
      return text_outputs

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))



  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example

      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]

      print("Loaded {} eval examples.".format(len(self.eval_data)))



  def evaluate(self, session):
    self.load_eval_data()

    tp,fn,fp = 0,0,0

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      top_span_starts, top_span_ends = session.run(self.predictions, feed_dict=feed_dict)


      gold_mentions = set([(m[0], m[1]) for cl in example["clusters"] for m in cl])
      pred_mentions = set([(s,e) for s,e in zip(top_span_starts,top_span_ends)])


      tp += len(gold_mentions & pred_mentions)
      fn += len(gold_mentions - pred_mentions)
      fp += len(pred_mentions - gold_mentions)


      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    m_r = float(tp)/(tp+fn)
    m_p = float(tp)/(tp+fp)
    m_f1 = 2.0*m_r*m_p/(m_r+m_p)

    print("Mention F1: {:.2f}%".format(m_f1*100))
    print("Mention recall: {:.2f}%".format(m_r*100))
    print("Mention precision: {:.2f}%".format(m_p*100))

    summary_dict = {}
    summary_dict["Mention F1"] = m_f1
    summary_dict["Mention recall"] = m_r
    summary_dict["Mention precision"] = m_p

    return util.make_summary(summary_dict), m_r