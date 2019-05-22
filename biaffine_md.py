from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util
from nn_md import NNMD


class BiaffineMD(NNMD):

  def add_model_specific_valuables(self, config):
    if config["lm_path"]:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
    else:
      self.lm_file = None
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]
    self.eval_data = None # Load eval data lazily.

    input_props = []
    input_props.append((tf.string, [None, None])) # Tokens.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings.
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    return input_props


  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")
    if not file_key in self.lm_file and file_key[:-2] in self.lm_file:
      file_key = file_key[:-2]
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb



  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = util.flatten(example["speakers"])

    assert num_words == len(speakers)

    max_sentence_length = max(len(s) for s in sentences)
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]

    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word

    tokens = np.array(tokens)

    doc_key = example["doc_key"]

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    lm_emb = self.load_lm_embeddings(doc_key)

    example_tensors = (tokens, lm_emb, text_len, is_training, gold_starts, gold_ends)

    if is_training and len(sentences) > self.config["max_training_sentences"]:
      return self.truncate_example(*example_tensors)
    else:
      return example_tensors

  def truncate_example(self, tokens,  lm_emb, text_len, is_training, gold_starts, gold_ends):
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = tokens.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    tokens = tokens[sentence_offset:sentence_offset + max_training_sentences, :]
    lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset

    return tokens, lm_emb, text_len, is_training, gold_starts, gold_ends



  def get_predictions_and_loss(self, inputs):
    tokens, lm_emb, text_len, is_training, gold_starts, gold_ends = inputs
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(tokens)[0]
    max_sentence_length = tf.shape(tokens)[1]

    if not self.lm_file:
      elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
      lm_embeddings = elmo_module(
          inputs={"tokens": tokens, "sequence_len": text_len},
          signature="tokens", as_dict=True)
      word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
      lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                         lm_embeddings["lstm_outputs1"],
                         lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
    lm_emb_size = util.shape(lm_emb, 2)
    lm_num_layers = util.shape(lm_emb, 3)
    with tf.variable_scope("lm_aggregation"):
      self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
      self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
    flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
    aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
    aggregated_lm_emb *= self.lm_scaling


    context_emb = aggregated_lm_emb

    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    num_words = tf.reduce_sum(text_len)
    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
                               [1, max_sentence_length])  # [num_sentences, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                               [1, max_sentence_length])  # [num_words, max_sentence_length]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(max_sentence_length), 0)  # [num_words, max_sentence_length]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                 candidate_starts)  # [num_words, max_sentence_length]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices,
                                               tf.minimum(candidate_ends, num_words - 1))  # [num_words, max_sentence_length]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                         candidate_end_sentence_indices))  # [num_words, max_sentence_length]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_sentence_length]

    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask)  # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]


    candidate_labels = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends)  # [num_candidates]

    candidate_scores_mask = tf.logical_and(tf.expand_dims(text_len_mask,[1]),tf.expand_dims(text_len_mask,[2])) #[num_sentence, max_sentence_length,max_sentence_length]
    sentence_ends_leq_starts = tf.tile(tf.expand_dims(tf.logical_not(tf.sequence_mask(tf.range(max_sentence_length),max_sentence_length)), 0),[num_sentences,1,1]) #[num_sentence, max_sentence_length,max_sentence_length]
    candidate_scores_mask = tf.logical_and(candidate_scores_mask,sentence_ends_leq_starts)

    flattened_candidate_scores_mask = tf.reshape(candidate_scores_mask,[-1]) #[num_sentence * max_sentence_length * max_sentence_length]


    context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask,self.lstm_dropout,False) # [num_sentence, max_sentence_length, emb]


    with tf.variable_scope("candidate_starts_ffnn"):
      candidate_starts_emb = util.projection(context_outputs,self.config["ffnn_size"]) #[num_sentences, max_sentences_length,emb]
    with tf.variable_scope("candidate_ends_ffnn"):
      candidate_ends_emb = util.projection(context_outputs,self.config["ffnn_size"]) #[num_sentences, max_sentences_length, emb]

    candidate_mention_scores = util.bilinear_classifier(candidate_starts_emb,candidate_ends_emb,self.dropout)#[num_sentence, max_sentence_length,max_sentence_length]
    candidate_mention_scores = tf.boolean_mask(tf.reshape(candidate_mention_scores,[-1]),flattened_candidate_scores_mask)

    loss = self.sigmoid_loss(candidate_mention_scores, candidate_labels)
    top_span_starts, top_span_ends = self.get_top_mentions(num_words,candidate_starts,candidate_ends,candidate_mention_scores)

    return [top_span_starts, top_span_ends], loss

