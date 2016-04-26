#!/usr/bin/python
# -*- coding: UTF-8 -*-

import re
import sys
import time
import jieba
import datetime
import os
import string
import math
from collections import Counter
reload(sys)
sys.setdefaultencoding('utf-8')   
def add_two_dim_dict(thedict,key1,key2,val):
    if(key1 in thedict):
        thedict[key1].update({key2:val})
    else:
        thedict.update({key1:{key2:val}})
def feature1(dic1,i,j):
    if(dic1[i][j]>=0.4):
        return 1
    else:
        return 0
        
def feature2(dic2,i,j):
    if(dic2[i][j]>=0.4):
        return 1
    else:
        return 0
def feature3(dic3,i,j):
    if(dic3[i][j]>=0.4):
        return 1
    else:
        return 0
def feature4(dic4,content):
    for i in dic4:
        if(dic4[i]==content[0]):
            return 1
    return 0
def feature5(content):
    for i in range(len(content)):
        if((content[i]=='0')or(content[i]=='1')or(content[i]=='2')or(content[i]=='3')or(content[i]=='4')or(content[i]=='5')or(content[i]=='6')or(content[i]=='7')or(content[i]=='8')or(content[i]=='9')):
            return 1
    return 0
   
time1=datetime.datetime.now()
seg_list=[]
total_list=[]
file1=open("BosonNLP_NER_6C.txt")
file2=open("Chinese-stop-words.txt")

#产生停止词列表
stop_list=[]
for line in file2:
    #print line
    length=len(line)
    content=unicode(line[0:len(line)],'gbk')
    #content_utf8=content.encode('utf-8')
    #print type(content)
    stop_list.append(content[0:len(content)-1])
file2.close()
stop_list.sort()




total_dic1={}
total_dic2={}
test_dic1={}
test_dic2={}
num1=0
num3=0
dic3={}#用于存储已知字典中的标签
#将训练数据进行切分
for line1 in file1:
    if(num1<1800):
        num2=0
    #total_list=[]
        dic1={}#用于存储上下文关系
        dic2={}#用于存储已经打上标签的词语
        line2=line1.split('{{')
        for line3 in line2:
            line4=line3.split('}}')
            if(len(line4)>1):
            #print unicode(line4[0],'utf-8')
                for i in range(len(unicode(line4[0],'utf-8'))):
                    if(unicode(line4[0],'utf-8')[i]==':'):
                    #print unicode(line4[0],'utf-8')[i+1:len(unicode(line4[0],'utf-8'))]
                        dic2[num2]=unicode(line4[0],'utf-8')[0:i]
                        dic1[num2]=unicode(line4[0],'utf-8')[i+1:len(unicode(line4[0],'utf-8'))]
                        if(dic1[num2] in dic3):
                            if(dic2[num2] in dic3[dic1[num2]]):
                                val=dic3[dic1[num2]][dic2[num2]]
                                add_two_dim_dict(dic3,dic1[num2],dic2[num2],val+1)
                            else:
                                add_two_dim_dict(dic3,dic1[num2],dic2[num2],1)
                        else:
                            add_two_dim_dict(dic3,dic1[num2],dic2[num2],1)
                    
                        num2+=1
                        break
                    seg_list=jieba.cut(unicode(line4[1],'utf-8'))
                    for seg in seg_list:
                        if(seg not in stop_list):
                            dic1[num2]=seg
                    #print seg
                            num2+=1
        
            else:
                seg_list=jieba.cut(unicode(line4[0],'utf-8'))
                for seg in seg_list:
                    if(seg not in stop_list):
                        dic1[num2]=seg
                    #print seg.flag
                    #print seg
                        num2+=1
            #print unicode(line4[0],'utf-8')
        total_dic1[num1]=dic1
        total_dic2[num1]=dic2
    else:
        num2=0
    #total_list=[]
        dic1={}#用于存储上下文关系
        dic2={}#用于存储已经打上标签的词语
        line2=line1.split('{{')
        for line3 in line2:
            line4=line3.split('}}')
            if(len(line4)>1):
            #print unicode(line4[0],'utf-8')
                for i in range(len(unicode(line4[0],'utf-8'))):
                    if(unicode(line4[0],'utf-8')[i]==':'):
                    #print unicode(line4[0],'utf-8')[i+1:len(unicode(line4[0],'utf-8'))]
                        dic2[num2]=unicode(line4[0],'utf-8')[0:i]
                        dic1[num2]=unicode(line4[0],'utf-8')[i+1:len(unicode(line4[0],'utf-8'))]
                        
                    
                        num2+=1
                        break
                    seg_list=jieba.cut(unicode(line4[1],'utf-8'))
                    for seg in seg_list:
                        if(seg not in stop_list):
                            dic1[num2]=seg
                    #print seg
                            num2+=1
        
            else:
                seg_list=jieba.cut(unicode(line4[0],'utf-8'))
                for seg in seg_list:
                    if(seg not in stop_list):
                        dic1[num2]=seg
                    #print seg.flag
                    #print seg
                        num2+=1
            #print unicode(line4[0],'utf-8')
        test_dic1[num3]=dic1
        test_dic2[num3]=dic2
        num3+=1
    num1+=1
    #print num1
file1.close()
#print test_dic2
dic1={}#用来记录在前面词语出现的条件下，命名实体出现的概率
dic2={}#用来记录在后面词语出现的条件下，命名实体出现的概率


#print total_dic3
for i in total_dic2:
    for j in total_dic2[i]:
        if(j!=0):
            if(total_dic1[i][j-1] in dic1):
                if(total_dic2[i][j] in dic1[total_dic1[i][j-1]]):
                    val=dic1[total_dic1[i][j-1]][total_dic2[i][j]]
                    add_two_dim_dict(dic1,total_dic1[i][j-1],total_dic2[i][j],val+1)
                else:
                    add_two_dim_dict(dic1,total_dic1[i][j-1],total_dic2[i][j],1)
            else:
                add_two_dim_dict(dic1,total_dic1[i][j-1],total_dic2[i][j],1)



for i in total_dic2:
    for j in total_dic2[i]:
        if((j+1) in total_dic1[i]):
            if(total_dic1[i][j+1] in dic2):
                if(total_dic2[i][j] in dic2[total_dic1[i][j+1]]):
                    val=dic2[total_dic1[i][j+1]][total_dic2[i][j]]
                    add_two_dim_dict(dic2,total_dic1[i][j+1],total_dic2[i][j],val+1)
                else:
                    add_two_dim_dict(dic2,total_dic1[i][j+1],total_dic2[i][j],1)
            else:
                add_two_dim_dict(dic2,total_dic1[i][j+1],total_dic2[i][j],1)

   
#求出前向条件概率，特征1      
for key1 in dic1:
    sum1=0
    for key2 in dic1[key1]:
        sum1+=dic1[key1][key2]
    for key2 in dic1[key1]:
        dic1[key1][key2]=(dic1[key1][key2]+0.0)/sum1
        #print key1,key2,dic1[key1][key2]      
#求出后向条件概率，特征2    
for key1 in dic2:
    sum1=0
    for key2 in dic2[key1]:
        sum1+=dic2[key1][key2]
    for key2 in dic2[key1]:
        dic2[key1][key2]=(dic2[key1][key2]+0.0)/sum1
        #print key1,key2,dic2[key1][key2]   
        
#在已知字典中是否出现，特征3
for key1 in dic3:
    sum1=0
    for key2 in dic3[key1]:
        sum1+=dic3[key1][key2]
    for key2 in dic3[key1]:
        dic3[key1][key2]=(dic3[key1][key2]+0.0)/sum1
        #print key1,key2,dic3[key1][key2]   
#第一个字是否在百家姓出现过，特征4
file3=open("name.txt")
dic4={}#存放百家姓
num=0
for line in file3:
    content=unicode(line[0:len(line)],'utf-8')
    list1=content[0:len(content)-1].split(' ')
    for name in list1:
        dic4[num]=name
        num+=1
        #print name
file3.close()
#是否含有数字，特征5,这里封装成函数
        
#是否含有地名，特征6
        
#是否含有组织机构名称，特征7
        



#MEMM模型,使用特征1，2，3,4，5

#初始化，第一步
sum1=0
sum2=0
sum3=0
sum4=0
sum5=0
num=0
F={}#特征的集合
f={}#第1位为s',第2位为O，第3位为S，第4位为序号，第5位为特征
for i in total_dic1:
    #F[i]={}
    for j in total_dic1[i]:
        
        if(j in total_dic2[i]):
            s=total_dic2[i][j]
        else:
            s=u'nothing'
        
        if((j-1) in total_dic2[i]):
            s_t=total_dic2[i][j-1]
        else:
            s_t=u'nothing'
        #print type(s_t)
        if(s_t not in f):
            f[s_t]={}
        if(total_dic1[i][j] not in f[s_t]):
            f[s_t][total_dic1[i][j]]={}
        if(s==u'nothing'):
            continue
        if(s not in f[s_t][total_dic1[i][j]]):
            f[s_t][total_dic1[i][j]][s]={}
        #if(num)
        
        #add_two_dim_dict(f,s_t,total_dic1[i][j],{})
        
        #add_two_dim_dict(f[s_t],total_dic1[i][j],s,num)
        
        
        '''
            
        if(total_dic1[i][j] in f):
            if(s in f[total_dic1[i][j]]):
                val=f[total_dic1[i][j]][s]
                add_two_dim_dict(f,total_dic1[i][j],s,num)
            else:
                add_two_dim_dict(f,total_dic1[i][j],s,num)
        else:
            add_two_dim_dict(f,total_dic1[i][j],s,num)
        '''
        f[s_t][total_dic1[i][j]][s][num]={}
        if((j-1) in total_dic1[i]):
            if(j in total_dic2[i]):
                f[s_t][total_dic1[i][j]][s][num][1]=feature1(dic1,total_dic1[i][j-1],total_dic2[i][j])
            else:
                f[s_t][total_dic1[i][j]][s][num][1]=0
        else:
            f[s_t][total_dic1[i][j]][s][num][1]=0
        if((j+1) in total_dic1[i]):
            if(j in total_dic2[i]):
                f[s_t][total_dic1[i][j]][s][num][2]=feature2(dic2,total_dic1[i][j+1],total_dic2[i][j])
            else:
                f[s_t][total_dic1[i][j]][s][num][2]=0
        else:
            f[s_t][total_dic1[i][j]][s][num][2]=0
        if(j in total_dic2[i]):
            f[s_t][total_dic1[i][j]][s][num][3]=feature3(dic3,total_dic1[i][j],total_dic2[i][j])
        else:
            f[s_t][total_dic1[i][j]][s][num][3]=0
            
        f[s_t][total_dic1[i][j]][s][num][4]=feature4(dic4,total_dic1[i][j])
        f[s_t][total_dic1[i][j]][s][num][5]=feature5(total_dic1[i][j])
        sum1+=f[s_t][total_dic1[i][j]][s][num][1]
        sum2+=f[s_t][total_dic1[i][j]][s][num][2]
        sum3+=f[s_t][total_dic1[i][j]][s][num][3]
        sum4+=f[s_t][total_dic1[i][j]][s][num][4]
        sum5+=f[s_t][total_dic1[i][j]][s][num][5]
        num+=1
        
F_a={}
F_a[1]=(sum1+0.0)/num
F_a[2]=(sum2+0.0)/num
F_a[3]=(sum3+0.0)/num
F_a[4]=(sum4+0.0)/num
F_a[5]=(sum5+0.0)/num  
#第二步     
a={}
a[0]={}
for j in range(num):
    a[0][j]={}
    a[0][j][1]=1
    a[0][j][2]=1
    a[0][j][3]=1
    a[0][j][4]=1
    a[0][j][5]=1

#第三步
for i in range(10):#最外面的大循环
    a[i+1]={}
    for j in f:#s'
        E={}   
        for n in range(1,6):
            E[n]=0.0001
        num1=0
        for k in f[j]:#O
            temp={}
            for n in range(1,6):
                temp[n]=0
                
            sum2=0
            sum0={}
            for t in f[j][k]:#s
                sum1=0
                
                for m in f[j][k][t]:
                    for n in range(1,6):
                        sum1+=a[i][m][n]*f[j][k][t][m][n]
                        #print f[j][k][t][m][n]
                if(sum1>=6):
                    sum1=6
                sum2+=math.e**sum1
                #print sum1
                #print sum1,sum2
                sum0[t]=math.e**sum1
            if(sum2!=0):
                for t in sum0:
                    sum0[t]=(sum0[t]+0.0)/sum2
            #else:
                
            sum2=0
            for t in f[j][k]:#s
                for m in f[j][k][t]:
                    for n in range(1,6):
                        temp[n]+=sum0[t]*f[j][k][t][m][n]
            
            for n in range(1,6):
                E[n]+=temp[n]
            num1+=1
        
        for n in range(1,6):
            E[n]=(E[n]+0.0)/num1
        #第3步结束
    #第4步
    #print num
    for j in range(num):
        a[i+1][j]={}
        for k in range(1,6):
            #print a[i][j][k]
            #print E[k]
            a[i+1][j][k]=a[i][j][k]+math.log(F_a[k]/E[k])/10
            #print a[i+1][j][k]
        #print a[i+1][j]
i=i+1
P={}   
for j in f:#s'
    P[j]={}
    for k in f[j]:#o
        P[j][k]={}
        sum2=0
        #temp={}
        for t in f[j][k]:#s
            sum1=0
            for m in f[j][k][t]:
                for n in range(1,6):
                    #print a[i][m][n]
                    sum1+=a[i][m][n]*f[j][k][t][m][n]
            if(sum1>=6):
                sum1=6
            sum2+=math.e**sum1
            P[j][k][t]=math.e**sum1
            #print sum1
        for t in f[j][k]:
            if(sum2!=0):
                #print P[j][k][t],sum2
                P[j][k][t]=(P[j][k][t]+0.0)/sum2
            else:
                 P[j][k][t]=0
            #print P[j][k][t]
            
        #print sum2
            

#对模型进行测试
#print P
sum1=0
sum2=0

for i in test_dic1:
    for j in test_dic1[i]:
        
        if(test_dic1[i][j] in P[u'nothing']):
            max1=0
            s_max1=u'product_name'
            s_max2=u''
            for t in P[u'nothing'][test_dic1[i][j]]:
                if(P[u'nothing'][test_dic1[i][j]][t]>=max1):
                    s_max1=t
                    max1=P[u'nothing'][test_dic1[i][j]][t]
            if(j in test_dic2[i]):
                s_max2=test_dic2[i][j]
                if(s_max1==s_max2):
                    sum1+=1
                else:
                    sum2+=1
                #print s_max1,s_max2
        elif(test_dic1[i][j] in P[u'product_name']):
            max1=0
            s_max1=u'product_name'
            s_max2=u''
            for t in P[u'product_name'][test_dic1[i][j]]:
                if(P[u'product_name'][test_dic1[i][j]][t]>=max1):
                    s_max1=t
                    max1=P[u'product_name'][test_dic1[i][j]][t]
            if(j in test_dic2[i]):
                s_max2=test_dic2[i][j]
                if(s_max1==s_max2):
                    sum1+=1
                else:
                    sum2+=1
                #print s_max1,s_max2
            #print s_max1,s_max2
        elif(test_dic1[i][j] in P[u'person_name']):
            max1=0
            s_max1=u'product_name'
            s_max2=u''
            for t in P[u'person_name'][test_dic1[i][j]]:
                if(P[u'person_name'][test_dic1[i][j]][t]>=max1):
                    s_max1=t
                    max1=P[u'person_name'][test_dic1[i][j]][t]
            if(j in test_dic2[i]):
                s_max2=test_dic2[i][j]
                if(s_max1==s_max2):
                    sum1+=1
                else:
                    sum2+=1
                #print s_max1,s_max2
        elif(test_dic1[i][j] in P[u'time']):
            max1=0
            s_max1=u'product_name'
            s_max2=u''
            for t in P[u'time'][test_dic1[i][j]]:
                if(P[u'time'][test_dic1[i][j]][t]>=max1):
                    s_max1=t
                    max1=P[u'time'][test_dic1[i][j]][t]
            if(j in test_dic2[i]):
                s_max2=test_dic2[i][j]
                if(s_max1==s_max2):
                    sum1+=1
                else:
                    sum2+=1
                #print s_max1,s_max2
                
        elif(test_dic1[i][j] in P[u'location']):
            max1=0
            s_max1=u'product_name'
            s_max2=u''
            for t in P[u'location'][test_dic1[i][j]]:
                if(P[u'location'][test_dic1[i][j]][t]>=max1):
                    s_max1=t
                    max1=P[u'location'][test_dic1[i][j]][t]
            if(j in test_dic2[i]):
                s_max2=test_dic2[i][j]
                if(s_max1==s_max2):
                    sum1+=1
                else:
                    sum2+=1
                #print s_max1,s_max2
        elif(test_dic1[i][j] in P[u'company_name']):
            max1=0
            s_max1=u'product_name'
            s_max2=u''
            for t in P[u'company_name'][test_dic1[i][j]]:
                if(P[u'company_name'][test_dic1[i][j]][t]>=max1):
                    s_max1=t
                    max1=P[u'company_name'][test_dic1[i][j]][t]
            if(j in test_dic2[i]):
                s_max2=test_dic2[i][j]
                if(s_max1==s_max2):
                    sum1+=1
                else:
                    sum2+=1
                #print s_max1,s_max2
        elif(test_dic1[i][j] in P[u'org_name']):
            max1=0
            s_max1=u'product_name'
            s_max2=u''
            for t in P[u'org_name'][test_dic1[i][j]]:
                if(P[u'org_name'][test_dic1[i][j]][t]>=max1):
                    s_max1=t
                    max1=P[u'org_name'][test_dic1[i][j]][t]
            if(j in test_dic2[i]):
                s_max2=test_dic2[i][j]
                if(s_max1==s_max2):
                    sum1+=1
                else:
                    sum2+=1
                #print s_max1,s_max2
        
print "准确率为：%f"%((sum1+0.0)/(sum1+sum2))
        
time2=datetime.datetime.now()
print u"运行时间%d秒"%(time2-time1).seconds
time.sleep(10)       