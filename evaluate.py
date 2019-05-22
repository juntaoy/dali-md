#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import util

if __name__ == "__main__":
  config = util.initialize_from_env()

  config['eval_path'] = config['test_path']
  if config['model_type'] in ['lee','biaffine']:
    config['lm_path'] = config['test_lm_path']

  model = util.get_model(config)
  with tf.Session() as session:
    model.restore(session)
    model.evaluate(session)
