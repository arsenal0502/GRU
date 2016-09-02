# -*- coding: UTF-8 -*-
import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample,select

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel


class Config(object):
  batch_size = 64
  embed_size = 70
  hidden_size = 220
  num_steps = 10
  max_epochs = 250
  early_stopping = 2
  dropout = 0.9
  lr = 0.001

class LSTM_Model(LanguageModel):

  def load_data(self, debug=False):
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    #self.encoded_test = np.array(
        #[self.vocab.encode(word) for word in get_ptb_dataset('test')],
        #dtype=np.int32)
    if debug:
      num_debug = 1024
      self.encoded_train = self.encoded_train[:num_debug]#读入训练数据
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]

  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, (None, self.config.num_steps))
    self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.num_steps))
    self.dropout_placeholder = tf.placeholder(tf.float32)
  
  def add_embedding(self):#将one-hot转化为词向量
    inputs = []
    with tf.device('/cpu:0'):
      L = tf.get_variable("Embedding", (len(self.vocab), self.config.embed_size))
      tensors = tf.nn.embedding_lookup(L, self.input_placeholder)
      split_tensors = tf.split(1, self.config.num_steps, tensors)
    
      for tensor in split_tensors:

        inputs.append(tf.squeeze(tensor, [1]))
      return inputs#返回的是一个list

  def add_projection(self, rnn_outputs):#把隐藏层转化为词语
    with tf.variable_scope("projection"):
      U=tf.get_variable("U",shape=(self.config.hidden_size,len(self.vocab)))
      b=tf.get_variable("b",shape=(len(self.vocab),))
    outputs=[tf.matmul(x,U)+b for x in rnn_outputs]###softmax?
    return outputs

  def add_loss_op(self, output):#计算损失函数
    loss = sequence_loss([output], [tf.reshape(self.labels_placeholder, [-1])], [tf.ones([self.config.batch_size * self.config.num_steps])])
    return loss

  def add_training_op(self, loss):#对损失函数进行优化
    optimizer=tf.train.AdamOptimizer(self.config.lr)
    train_op=optimizer.minimize(loss)
    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.lstm_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.lstm_outputs)

    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)


  def add_model(self, inputs):
    
    hidden_size=self.config.hidden_size
    embed_size=self.config.embed_size
    batch_size=self.config.batch_size
    with tf.variable_scope("LSTM"):
      U_i=tf.get_variable("U_i",shape=(hidden_size,hidden_size))
      U_f=tf.get_variable("U_f",shape=(hidden_size,hidden_size))
      U_o=tf.get_variable("U_o",shape=(hidden_size,hidden_size))
      U_c=tf.get_variable("U_c",shape=(hidden_size,hidden_size))
      W_i=tf.get_variable("W_i",shape=(embed_size,hidden_size))
      W_f=tf.get_variable("W_f",shape=(embed_size,hidden_size))
      W_o=tf.get_variable("W_o",shape=(embed_size,hidden_size))
      W_c=tf.get_variable("W_c",shape=(embed_size,hidden_size))
      
    self.initial_state=tf.zeros([batch_size,hidden_size*2])
    pre_h,pre_c=tf.split(1,2,self.initial_state)
    
    lstm_outputs=[]
    for step in inputs:
      step=tf.nn.dropout(step,self.dropout_placeholder)
      i_t=tf.sigmoid(tf.matmul(step,W_i)+tf.matmul(pre_h,U_i))
      f_t=tf.sigmoid(tf.matmul(step,W_f)+tf.matmul(pre_h,U_f))
      o_t=tf.sigmoid(tf.matmul(step,W_o)+tf.matmul(pre_h,U_o))
      c_t=tf.tanh(tf.matmul(step,W_c)+tf.matmul(pre_h,U_c))
      pre_c=f_t*pre_c+i_t*c_t
      pre_h=o_t*tf.tanh(pre_c)
      lstm_outputs.append(tf.nn.dropout(pre_h,self.dropout_placeholder))
    self.final_state=tf.concat(1,[pre_h,pre_c])
    return lstm_outputs


  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))#总的迭代次数
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',next_text='<eos>',
                  stop_length=100, temp=1.0):

  state = model.initial_state.eval()
  list1=[]
  stop_tokens=['<eos>']
  tokens = [model.vocab.encode(word) for word in starting_text]#找到starting_text中单词的索引
  for i in range(len(starting_text)):
    token_list = np.array(tokens[i]).reshape((1, model.config.num_steps))#最近的结果作为下一次的输入
    feed = {model.input_placeholder: token_list,
              model.initial_state: state,
              model.dropout_placeholder: 1}
    y_pred, state = session.run(
          [model.predictions[-1], model.final_state], feed_dict=feed)#model.predictions[-1]为模型前一个的预测结果
    for j in range(i+1,len(starting_text)):
      b=tokens[j]
      pro= select(y_pred[0],b, temperature=temp)
      #print pro
      list1.append(pro)
  return list1

def generate_sentence(session, model, config, *args1, **kwargs):
  return generate_text(session, model, config, *args1, stop_tokens=['<eos>'], **kwargs)

def test_LSTM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1
  with tf.variable_scope('LSTMLM') as scope:
    model = LSTM_Model(config)
    scope.reuse_variables()
    gen_model = LSTM_Model(gen_config)

  init = tf.initialize_all_variables()#将变量都初始化
  saver = tf.train.Saver()#创建一个 Saver 来管理模型中的所有变量

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
    
    session.run(init)
    for epoch in xrange(config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,#参数model.encode_train是单词索引的list
          train_op=model.train_step)
      valid_pp = model.run_epoch(session, model.encoded_valid)
      print 'Training perplexity: {}'.format(train_pp)
      print 'Validation perplexity: {}'.format(valid_pp)
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
      saver.save(session, './ptb_rnnlm.weights')#将权重参数保存
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
    
    saver.restore(session, 'ptb_rnnlm.weights')#读取权重参数
    file1=open("output_LSTM.csv","wb")
    file2=open("data/train.txt")
    Type="Directed"
    Id=2000
    for line in file2:
      split=line.split(" ")
      if(len(split)==0):
        continue
      list2=[]
      for l in split:
        if(l=="\n"):
          continue
        list2.append(l)
      pro=generate_text(session, gen_model, gen_config, starting_text=list2,next_text=list2,  temp=1.0)#返回的pro是一个list
      k=0
      for i in range(len(list2)):
        for j in range(i+1,len(list2)):
          file1.write("{} {} {} ".format(list2[i],list2[j],pro[k]))
          k+=1
      file1.write("\n")
          
      
    file1.close()

if __name__ == "__main__":
    test_LSTM()
