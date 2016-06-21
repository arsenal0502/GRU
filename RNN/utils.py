# -*- coding: UTF-8 -*-
from collections import defaultdict

import numpy as np

class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<unk>'
    self.add_word(self.unknown, count=0)#将<unk>加入字典中

  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):#将单词转化成索引
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):#将索引转化成单词
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)

def calculate_perplexity(log_probs):#计算混乱度
  # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
  perp = 0
  for p in log_probs:
    perp += -p
  return np.exp(perp / len(log_probs))

def get_ptb_dataset(dataset='train'):#读入数据
  fn = 'data/{}.txt'
  #fn = 'data/ptb/ptb.{}.txt'
  for line in open(fn.format(dataset)):
    for word in line.split():
      yield word
    # Add token to the end of the line
    # Equivalent to <eos> in:
    # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31
    yield '<eos>'

def ptb_iterator(raw_data, batch_size, num_steps):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]#data[i]为第i行数据
  epoch_size = (batch_len - 1) // num_steps
  #print batch_len
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = data[:, i * num_steps:(i + 1) * num_steps]#10个1组
    y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]#y是x向后偏移一位
    #print x,y
    yield (x, y)

def select(a, b,temperature=1.0):#采样,选出概率最大的值,b是下一个的索引
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    #举例：抛色子20次，结果分布：np.random.multinomial(20, [1/6.]*6, size=1)，array([[4, 1, 7, 5, 2, 1]])
    return a[b]#多项分布，为了引入随机性

def sample(a, temperature=1.0):#采样,选出概率最大的值
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    #举例：抛色子20次，结果分布：np.random.multinomial(20, [1/6.]*6, size=1)，array([[4, 1, 7, 5, 2, 1]])
    L=np.random.multinomial(1, a, 1)
    return np.argmax(L)#多项分布，为了引入随机性

def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
      y_indices = data_y[batch_start:batch_start + batch_size]
      y = np.zeros((len(x), label_size), dtype=np.int32)
      y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield x, y
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
