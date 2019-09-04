import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf


### TFRecords Functions ###
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_feature(value):
    """Returns a bytes_list from a list of strings."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))