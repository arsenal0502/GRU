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
  embed_size = 50
  hidden_size = 100
  num_steps = 10
  max_epochs = 10
  early_stopping = 2
  dropout = 0.9
  lr = 0.001

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
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
      b_2=tf.get_variable("b_2",shape=(len(self.vocab),))
    outputs=[tf.matmul(x,U)+b_2 for x in rnn_outputs]###softmax?
    

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
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)


  def add_model(self, inputs):
    
    hidden_size=self.config.hidden_size
    embed_size=self.config.embed_size
    batch_size=self.config.batch_size
    with tf.variable_scope("RNN"):
      H=tf.get_variable("H",shape=(hidden_size,hidden_size))
      I=tf.get_variable("I",shape=(embed_size,hidden_size))
      b_1=tf.get_variable("b_1",shape=(hidden_size,))
    self.initial_state=tf.zeros([batch_size,hidden_size])
    pre_h=self.initial_state
    rnn_outputs=[]
    for step in inputs:
      step=tf.nn.dropout(step,self.dropout_placeholder)
      pre_h=tf.sigmoid(tf.matmul(pre_h,H)+tf.matmul(step,I)+b_1)
      rnn_outputs.append(tf.nn.dropout(pre_h,self.dropout_placeholder))
    self.final_state=pre_h
    return rnn_outputs


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
  stop_tokens=['<eos>']
  tokens = [model.vocab.encode(word) for word in starting_text.split()]#找到starting_text中单词的索引

  token_list = np.array(tokens[-1]).reshape((1, model.config.num_steps))#最近的结果作为下一次的输入
  feed = {model.input_placeholder: token_list,
              model.initial_state: state,
              model.dropout_placeholder: 1}

  y_pred, state = session.run(
          [model.predictions[-1], model.final_state], feed_dict=feed)#model.predictions[-1]为模型前一个的预测结果
  b=model.vocab.encode(next_text)
  pro= select(y_pred[0],b, temperature=temp)
  return pro


def generate_sentence(session, model, config, *args1, **kwargs):
  return generate_text(session, model, config, *args1, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM_Model(config)
    scope.reuse_variables()
    gen_model = RNNLM_Model(gen_config)

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
      #valid_pp = model.run_epoch(session, model.encoded_valid)
      print 'Training perplexity: {}'.format(train_pp)
      #print 'Validation perplexity: {}'.format(valid_pp)
      #if valid_pp < best_val_pp:
      #  best_val_pp = valid_pp
      #  best_val_epoch = epoch
      saver.save(session, './ptb_rnnlm.weights')#将权重参数保存
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
    '''
    saver.restore(session, 'ptb_rnnlm.weights')#读取权重参数

    test_pp = model.run_epoch(session, model.encoded_test)#测试准确率
    print '=-=' * 5
    print 'Test perplexity: {}'.format(test_pp)
    print '=-=' * 5
    '''
    result=result_english()
    file1=open("output_RNN.csv","wb")
    file1.write("Source,Target,Type,Id,Weight\n")
    i=1
    starting_text = result[i].split(",")[0]
    next_text = result[i].split(",")[1]
    Type=result[i].split(",")[2]
    Id=result[i].split(",")[3]
    while starting_text:
      pro=generate_text(session, gen_model, gen_config, starting_text=starting_text,next_text=next_text,  temp=1.0)
      file1.write("{},{},{},{},{}\n".format(starting_text,next_text,Type,Id,pro))
      i+=1
      if(i in result):
        starting_text = result[i].split(",")[0]
        next_text = result[i].split(",")[1]
        Type=result[i].split(",")[2]
        Id=result[i].split(",")[3]
      else:
        break
    '''
    starting_text = raw_input('start> ')
    next_text = raw_input('next> ')
    while starting_text:
      generate_text(session, gen_model, gen_config, starting_text=starting_text,next_text=next_text,  temp=1.0)
      starting_text = raw_input('start> ')
      next_text = raw_input('next> ')
    '''
def result_RNN():
  file1=open("output.csv")
  i=0
  result={}
  for line in file1:
    if(i==0):
      i+=1
      continue
    else:
      split=line.split(",")
      result[i]=split[0]+','+split[1]+','+split[2]+','+split[3]
      i+=1
  file1.close()
  return result
def result_english():
  file1=open("data/train.txt")
  result={}
  i=1
  for line in file1:
    split=line.replace("\n","").split(" ")
    l=len(split)
    for j in range(l-1):
      str1="{},{},Directed,1".format(split[j],split[j+1])
      if(str1 not in result):
        result[str1]=i
        i+=1
  result={value:key for key,value in result.iteritems()}
  file1.close()
  return result
if __name__ == "__main__":
    test_RNNLM()
