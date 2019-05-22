from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import bert_tokenization

import util
from nn_md import NNMD

class BertMD(NNMD):
  def add_model_specific_valuables(self,config):
    self.bert_url = str(self.config["bert_url"])
    self.bert_size = self.config["bert_size"]
    self.max_bert_sent_len = self.config["max_bert_sent_len"]
    self.bert_tokenizer = self.load_bert_vocab()

    self.max_span_width = config["max_span_width"]
    self.eval_data = None  # Load eval data lazily.

    input_props = []
    input_props.append((tf.int32, [None, self.max_bert_sent_len]))  # Bert token ids.
    input_props.append((tf.int32, [None]))  # Text lengths.
    input_props.append((tf.int32, [None]))  # Bert text lengths. as it use wordpiece embeddings
    input_props.append((tf.bool, [None, None]))  # Bert sentences mask to retrain only the first wordpiece of the token
    input_props.append((tf.bool, []))  # Is training.
    input_props.append((tf.int32, [None]))  # Gold starts.
    input_props.append((tf.int32, [None]))  # Gold ends.
    return input_props

  def restore(self, session):
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)


  def load_bert_vocab(self):
    with tf.Graph().as_default():
      bert_model = hub.Module(self.bert_url)
      vocab_info = bert_model(signature="tokenization_info",as_dict=True)
      with tf.Session() as sess:
        vocab_file, do_lower_case = sess.run([vocab_info["vocab_file"],vocab_info["do_lower_case"]])

    return bert_tokenization.FullTokenizer(vocab_file=vocab_file,do_lower_case=do_lower_case)



  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))

    sentences = example["sentences"]

    text_len = np.array([len(s) for s in sentences])
    bert_index = np.zeros([len(sentences),self.max_bert_sent_len])
    bert_len = np.zeros([len(sentences)])
    bert_mask = np.zeros([len(sentences), self.max_bert_sent_len],dtype=np.bool)
    for i, sentence in enumerate(sentences):
      bert_token = []
      bert_token.append('[CLS]')
      for j, word in enumerate(sentence):
        bert_mask[i][len(bert_token)] = True
        bert_token.extend(self.bert_tokenizer.tokenize(word))
      bert_token.append('[SEP]')
      bert_len[i] = len(bert_token)
      bert_index[i][:len(bert_token)] = self.bert_tokenizer.convert_tokens_to_ids(bert_token)


    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    example_tensors = (bert_index, text_len, bert_len,bert_mask, is_training, gold_starts, gold_ends)

    if is_training and len(sentences) > self.config["max_training_sentences"]:
      return self.truncate_example(*example_tensors)
    else:
      return example_tensors

  def truncate_example(self, bert_index, text_len, bert_len, bert_mask,is_training, gold_starts, gold_ends):
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = text_len.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    bert_index = bert_index[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]
    bert_len = bert_len[sentence_offset:sentence_offset + max_training_sentences]
    bert_mask = bert_mask[sentence_offset:sentence_offset + max_training_sentences]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset

    return bert_index, text_len, bert_len,bert_mask,is_training, gold_starts, gold_ends



  def get_predictions_and_loss(self,  inputs):
    bert_index, text_len, bert_len, bert_token_mask, is_training, gold_starts, gold_ends = inputs
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

    num_sentences = tf.shape(text_len)[0]
    max_sentence_length = tf.reduce_max(text_len)


    bert_sent_mask = tf.sequence_mask(bert_len,self.max_bert_sent_len,tf.int32)
    bert_segment = tf.zeros_like(bert_sent_mask,tf.int32)

    bert_module = hub.Module(self.bert_url,trainable=True)
    bert_inputs = dict(input_ids=bert_index, input_mask=bert_sent_mask, segment_ids=bert_segment)
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens",as_dict=True)
    context_outputs = bert_outputs["sequence_output"] #[num_sentences, max_bert_sent_len, emb]


    context_outputs = self.flatten_emb_by_sentence(context_outputs,bert_token_mask) #[num_words,emb]


    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    num_words = tf.reduce_sum(text_len)


    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]

    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]

    candidate_labels = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends) # [num_candidates]


    span_start_emb = tf.gather(context_outputs, candidate_starts)  # [k, emb]
    span_end_emb = tf.gather(context_outputs, candidate_ends)  # [k, emb]

    candidate_span_emb = tf.concat([span_start_emb,span_end_emb],1)


    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb,self.dropout) # [k, 1]
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

    loss = self.sigmoid_loss(candidate_mention_scores,candidate_labels)

    top_span_starts, top_span_ends = self.get_top_mentions(num_words,candidate_starts,candidate_ends,candidate_mention_scores)

    return [top_span_starts, top_span_ends], loss








