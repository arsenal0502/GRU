# -*- coding: UTF-8 -*-
import numpy as np
import random
import sys
import random
import math
reload(sys)
sys.setdefaultencoding('utf-8')
def softmax(x):
    x=x.astype(float)
    if(len(x)==1):
        x=np.exp(x-x.max())
        x_sum=np.sum(x)
        x=x/x_sum
    else:
        if(type(x[0])!=type(x)):
            x=np.exp(x-x.max())
            x_sum=np.sum(x)
            x=x/x_sum
        else:
            for i in range(len(x)):
                x[i]=x[i]-x[i].max()
                x_exp=np.exp(x[i])
                x_sum=np.sum(x_exp)
                x[i]=(x_exp+0.0)/x_sum
    return x
def gradcheck_naive(f, x):#根据梯度定义对梯度进行检查
    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) 
    h = 1e-4


    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value=x[ix]
        x[ix]=old_value-h#只对某一方向进行变化
        random.setstate(rndstate) 
        fxsh=f(x)[0]
        x[ix]=old_value+h#只对某一方向进行变化
        random.setstate(rndstate) 
        fxph=f(x)[0]
        x[ix]=old_value#将这一方向恢复原值
        numgrad=(fxph-fxsh)/(2*h)



        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext() 

    print "Gradient check passed!"
def sigmoid(x):
    x=1/(1+np.exp(-x))    
    return x

def sigmoid_grad(f):
    f=f*(1-f)
    return f
def normalizeRows(x):
    
    sqrt=np.sqrt(np.sum(np.square(x),axis=1)).reshape(-1,1)#reshape(-1,1)将矩阵竖向立起来
    x=x/sqrt
    return x
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    i,j=outputVectors.shape
    scores=np.dot(outputVectors,predicted).reshape(1,i)
    result=softmax(scores).reshape(i,)
    cost=-np.log(result[target])
    labels=np.zeros(i)
    labels[target]=1#如果target发生改变，每次的labels都发生变化
    dscores = result - labels
    gradPred=np.dot(dscores,outputVectors)#1*i和i*j点乘
    grad = dscores.reshape(i,1).dot(predicted.reshape(1,-1)) #i*1和1*j点乘
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    i,j=outputVectors.shape
    sampleIndex=[]
    for index in range(K):
        currentIndex=dataset.sampleTokenIdx()
        sampleIndex.append(currentIndex)
    w_k=outputVectors[sampleIndex,:]
    w_i_r=sigmoid(np.dot(outputVectors[target],predicted))
    w_k_r=sigmoid(-np.dot(w_k,predicted))
    cost=-np.log(w_i_r)-np.sum(np.log(w_k_r))
    gradPred=outputVectors[target]*(w_i_r-1)+np.dot((1-w_k_r),w_k)
    grad=np.zeros(outputVectors.shape)
    grad[target]=predicted*(w_i_r-1)
    for index in range(K):
        grad[sampleIndex[index]]+=predicted*(1-w_k_r)[index]
    return cost, gradPred, grad
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    predicted=inputVectors[tokens[currentWord]]
    cost=0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for word in contextWords:
        
        target=tokens[word]
        current_cost,current_gradIn,current_gradOut=word2vecCostAndGradient(predicted, target, outputVectors,dataset)
        cost+=current_cost#对每一次target都求一次交叉熵
        gradIn[tokens[currentWord]]+=current_gradIn#结果是只有一列非零的矩阵
        gradOut+=current_gradOut#每次target不同,所以将所得矩阵累加
    #gradIn,gradOut都是矩阵
    
    return cost, gradIn, gradOut
def test_word2vec():
    #dummy_tokens= dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    filename="test.txt"
    dummy_tokens , _=doc_to_dict_and_list(filename)
    num=len(dummy_tokens)
    x=num*2#gradin,gradout的行数
    y=3#gradin,gradout的列数
    lr=0.1#学习率
    windows=1
    find_centre(filename,windows)
    
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, num-1)

    def getRandomContext(C):
        _ ,tokens =doc_to_dict_and_list("test.txt")
        return tokens[random.randint(0,num-1)], [tokens[random.randint(0,num-1)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(x,y))#1-5行为输入向量，6-10行为输出向量
    print "\n=== Results ==="
    pre_cost=10000
    list_centre,list_around=find_centre(filename,windows)
    while(1):
      ###cost,gradin,gradout=skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:num,:], dummy_vectors[num:,:], dataset, negSamplingCostAndGradient)
      total_cost=0
      for i in range(len(list_centre)):
        cost,gradin,gradout=skipgram(list_centre[i], y, list_around[i], dummy_tokens, dummy_vectors[:num,:], dummy_vectors[num:,:], dataset)
        total_cost+=cost
        dummy_vectors[:num,:]=dummy_vectors[:num,:]-gradin*lr
        dummy_vectors[num:,:]=dummy_vectors[num:,:]-gradout*lr
      print total_cost
      if(abs(pre_cost-total_cost)<1e-5):
        break
      else:
        pre_cost=total_cost
    print dummy_vectors
#函数1：将句子中的单词转换成字典和list
def doc_to_dict_and_list(filename):
    file1=open(filename)
    dict_output={}
    list_output=[]
    num=0
    for line in file1:
      split=line.replace('\n','').split()#如果不是最后一行，要将每行最后的/n去掉
      for i in split:
        if(i not in dict_output):
          dict_output[i]=num
          list_output.append(i)
          num+=1
    file1.close()
    return dict_output,list_output
#函数2：找出中心词和周围词
def find_centre(filename,windows):
    file1=open(filename)
    list_centre=[]
    list_around=[]
    for line in file1:
      split=line.replace('/n','').split()#如果不是最后一行，要将每行最后的/n去掉
      for i in range(len(split)):
        if(i<windows):
          list_centre.append(split[i])
          list_output=[]
          list_output.append(split[i+1:i+windows+1][0])
          list_around.append(list_output)
        elif(i>=(len(split)-windows)):
          list_centre.append(split[i])
          list_output=[]
          list_output.append(split[i-windows:i][0])
          list_around.append(list_output)
        else:
          list_centre.append(split[i])
          list_output=[]
          list_output.append(split[i-windows:i][0])
          list_output.append(split[i+1:i+windows+1][0])
          list_around.append(list_output)
    return list_centre,list_around
          
    
if __name__ == "__main__":
    test_word2vec()
