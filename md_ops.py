from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

md_op_library = tf.load_op_library("./md_kernels.so")

extract_spans = md_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")
