# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from collections import defaultdict
import re
from textblob import TextBlob
import urllib
import ast
import math
from sklearn import preprocessing

# Importing the dataset
data = pd.read_csv('title_rel_assigned_Haley - title_rel_assigned.csv', engine = 'python', encoding = 'latin-1')

PMID = data['PMID']
PMID_list = list(PMID)

keys = open("keyword_corpus_imp_paper_2.txt", "r")
keys = ast.literal_eval(keys.read())
keys = set(keys)
keys.remove('')   
keys = list(keys)

keys_1_gram = []
keys_2_gram = []
keys_3_gram = []
keys_1_2_3_gram = []
for i in range(len(keys)):
    length = len(keys[i].split())
    if(length == 1):
        keys_1_2_3_gram.append(keys[i])
        keys_1_gram.append(keys[i])
    elif(length == 2):
        keys_1_2_3_gram.append(keys[i])
        keys_2_gram.append(keys[i])
    elif(length == 3):
        keys_1_2_3_gram.append(keys[i])
        keys_3_gram.append(keys[i])
    else:
        continue

#dataframe = data[1457: 1987]
dataframe = data[1887: 1987]

author1 = pd.DataFrame()
author1['PMID'] = dataframe.PMID
author1['paper_titles'] = dataframe['Unnamed: 4']

author1['assigned?'] = dataframe['assigned?']
author1['Haley'] = dataframe['Haley']
author1['Peng Yu'] = dataframe['Peng Yu']
author1['Karthik'] = dataframe['Karthik']
author1['Chenwei'] = dataframe['Chenwei']
author1['Xiaohuan'] = dataframe['Xiaohuan']
author1['Yi Weng'] = dataframe['Yi Weng']


# Preprocessing -> stopword removal and lower case for author dataframe
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

comment_dict = defaultdict(list)
for i in range(len(author1)):
    sentence = re.sub('[^a-zA-Z.]',' ',list(author1['paper_titles'])[i])
    sentence = sentence.lower()
    sentence = sentence.split('.')
    for k in range(len(sentence)):
        review = sentence[k].split()
        review = [word for word in review if not word in set(stopwords.words('english'))]
        sentence[k] =  ' '.join(review)
        comment_dict[i].append(sentence[k])
 
#delete unwanted '' words
for j in range(len(comment_dict)):
    comment_dict[j] = [comment_dict[j][i] for i in range(len(comment_dict[j])) if comment_dict[j][i] not in '']
paper_text = defaultdict(list)
for i in range(len(comment_dict)):
    paper_text[i] = ('. '.join(comment_dict[i][j] for j in range(len(comment_dict[i]))))
    
author1['paper_titles'] = [paper_text[i] for i in range(len(paper_text))]
corpus_list = defaultdict(list)
# single words       
for i in range(len(paper_text)):
    text = TextBlob(paper_text[i])
    text = text.ngrams(n=1)
    for k in range(len(text)):
        triword = [' '.join([text[k][l] for l in range(len(text[k]))])]
        triword = triword[0]
        corpus_list[i].append(triword)
# biwords       
for i in range(len(paper_text)):
    text = TextBlob(paper_text[i])
    text = text.ngrams(n=2)
    for k in range(len(text)):
        triword = [' '.join([text[k][l] for l in range(len(text[k]))])]
        triword = triword[0]
        corpus_list[i].append(triword)
# triwords       
for i in range(len(paper_text)):
    text = TextBlob(paper_text[i])
    text = text.ngrams(n=3)
    for k in range(len(text)):
        triword = [' '.join([text[k][l] for l in range(len(text[k]))])]
        triword = triword[0]
        corpus_list[i].append(triword)
        
author1['1,2,3 gram'] = [corpus_list[i] for i in range(len(corpus_list))]

# corpus contains all ngrams of paper visited.
corpus = set()
for i in range(len(corpus_list)):
    for j in range(len(corpus_list[i])):
        corpus.add(corpus_list[i][j])
corpus = list(sorted(corpus))

# Getting words and papers
words = defaultdict(list)
for i in range(len(corpus)):
    words[i] = corpus[i]
papers = list(author1['PMID'])

# see paper_df and word_df

word_df = pd.DataFrame()
word_df['word number'] = words.keys()
word_df['word name'] = words.values()
#word_id = word_df['word number'].values

word_to_number = defaultdict(list)
for i in words.keys():
    word_to_number[list(words.values())[i]].append(i)
    
paper_list = defaultdict(list)
paper_words = author1['1,2,3 gram']

for i in author1['1,2,3 gram'].keys():
    for j in range(len(paper_words[i])):
        paper_list[i].append(word_to_number[paper_words[i][j]])
        
paper_df = pd.DataFrame(index = paper_words.index)
paper_df['1,2,3 gram'] = [paper_list[i] for i in paper_list.keys()]
#paper_id = paper_df.values
author1['1,2,3 gram numbers'] = paper_df

author1_word_list = list(author1['1,2,3 gram'])
n_gram_keywords = defaultdict(set)
n_gram_no_keywords = defaultdict(set)
for i in range(len(author1_word_list)):
    for j in range(len(author1_word_list[i])):
        if(author1_word_list[i][j] in keys):
            n_gram_keywords[i].add('')
            n_gram_keywords[i].add(author1_word_list[i][j])
        else:
            n_gram_no_keywords[i].add(author1_word_list[i][j])
            n_gram_keywords[i].add('')

for i in n_gram_keywords:
    n_gram_keywords[i] = list(n_gram_keywords[i])[1:]
    
for i in n_gram_no_keywords:
    n_gram_no_keywords[i] = list(n_gram_no_keywords[i])

temp1 = []
for i in range(len(n_gram_keywords)):
    temp1.append(n_gram_keywords[i])

author1['1,2,3 gram keywords'] = temp1


temp2 = []
for i in range(len(temp1)):
    t=[]
    for j in range(len(temp1[i])):
        t.append(word_to_number[temp1[i][j]])
    temp2.append(t)

for i in range(len(temp2)):
    if(temp2[i] == []):
        continue
    else:
        for j in range(len(temp2[i])):
            if(temp2[i][j] == []):
                continue
            else:
                temp2[i][j] = temp2[i][j][0]  
author1['1,2,3 gram keywords number'] = temp2


temp3 = []
for i in range(len(n_gram_no_keywords)):
    temp3.append(n_gram_no_keywords[i])        
author1['1,2,3 gram No keywords'] = temp3


temp4 = []
for i in range(len(temp3)):
    t=[]
    for j in range(len(temp3[i])):
        t.append(word_to_number[temp3[i][j]])
    temp4.append(t)   
for i in range(len(temp4)):
    for j in range(len(temp4[i])):
        temp4[i][j] = temp4[i][j][0]
author1['1,2,3 gram No keywords number'] = temp4
#author1.set_index(author1.index, inplace = True)
author1 = author1.sort_index()

temp_list1 = []
temp_list1_PMID = []
temp_list2 = []
for i in author1.index:
    for j in range(len(author1['1,2,3 gram numbers'][i])):
        temp_list1.append(i)
        temp_list1_PMID.append(author1.PMID[i])
        temp_list2.append(author1['1,2,3 gram numbers'][i][j][0])

df = pd.DataFrame()
df['index'] = temp_list1
df['PMID'] = temp_list1_PMID
df['words_number'] = temp_list2

temp_index = []
temp_key = []
temp_PMID = []
temp_not_key = []
temp_words_number = []

for j in author1.index:
    for i in df.index:    
        if(df['words_number'][i] in author1['1,2,3 gram keywords number'][j]):
            temp_index.append(j)
            temp_PMID.append(author1['PMID'][j])
            temp_key.append(2)
            temp_words_number.append(df['words_number'][i])
        elif(df['words_number'][i] in author1['1,2,3 gram No keywords number'][j]):
            temp_index.append(j)
            temp_PMID.append(author1['PMID'][j])
            temp_key.append(1)
            temp_words_number.append(df['words_number'][i])
        else: 
            continue
    print(j)

train_test_dataframe = pd.DataFrame()
train_test_dataframe['Paper Index'] = temp_index
train_test_dataframe['PMID'] = temp_PMID
train_test_dataframe['Word ID'] = temp_words_number
train_test_dataframe['Key Rating'] = temp_key

train_test_dataframe.to_csv('final_train_test_dataframe.csv')

train_test_dataframe = pd.read_csv('final_train_test_dataframe.csv')
del train_test_dataframe['Unnamed: 0']
del train_test_dataframe['PMID']


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(train_test_dataframe, test_size = 0.5, random_state = 0)
                    
# Preparing the training set and the test set
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')
train_test_dataframe_set = np.array(train_test_dataframe, dtype = 'int')

# Getting the number of papers and words
nb_papers = int(max(len(training_set[:,0]), len(test_set[:,0])))
nb_words = int(max(len(training_set[:,1]), len(test_set[:,1])))

nb_papers = int(len(train_test_dataframe_set[:, 0]))
nb_words = int(len(train_test_dataframe_set[:, 1]))

my_papers_training = training_set[:,0]
my_words_training = training_set[:,1]
my_key_training = training_set[:,2]
my_dataframe_training = pd.DataFrame()
my_dataframe_training['Paper ID'] = my_papers_training
my_dataframe_training['Word ID'] = my_words_training
my_dataframe_training['Keyword yes(2)/No(1)'] = my_key_training

my_papers_test = test_set[:,0]
my_words_test = test_set[:,1]
my_key_test = test_set[:,2]
my_dataframe_test = pd.DataFrame()
my_dataframe_test['Paper ID'] = my_papers_test
my_dataframe_test['Word ID'] = my_words_test
my_dataframe_test['Keyword yes(2)/No(1)'] = my_key_test

my_papers = train_test_dataframe_set[:, 0]
my_words = train_test_dataframe_set[:, 1]
my_key = train_test_dataframe_set[:, 2]
my_dataframe = pd.DataFrame()
my_dataframe['Paper ID'] = my_papers
my_dataframe['Word ID'] = my_words
my_dataframe['Keyword yes(2)/No(1)'] = my_key

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_papers in author1.index:
        id_words = data[:,1][data[:,0] == id_papers]
        id_ratings = data[:,2][data[:,0] == id_papers]
        ratings = np.zeros(nb_words)
        ratings[id_words] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
train_test_dataframe_set = convert(train_test_dataframe_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
train_test_dataframe_set = torch.FloatTensor(train_test_dataframe_set)
# Converting the ratings into binary ratings 2 (Keywords present) or 1 (keywords not present)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 1
train_test_dataframe_set[test_set == 0] = -1
train_test_dataframe_set[test_set == 1] = 0
train_test_dataframe_set[test_set == 2] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W = self.W.t()
        self.W = self.W + torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.W = self.W.t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Getting the number of papers and words
nb_papers = int(max(len(training_set[:,0]), len(test_set[:,0])))
nb_words = int(max(len(training_set[:,1]), len(test_set[:,1])))

nb_papers = int(len(train_test_dataframe_set[:, 0]))
nb_words = int(len(train_test_dataframe_set[:, 1]))

nv = len(training_set[0])
nv = len(train_test_dataframe_set[0])

nh = 100
batch_size = 10
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_papers - batch_size, batch_size):
#        print(id_user)
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
#            print('inner', k)
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

temp = []
temp_s_l = []
s_count = []
# Testing the RBM
test_loss = 0
for id_user in range(nb_papers):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        v_list = v[vt>=0]
        vt_list = vt[vt>=0]
        s = 0.
        ss = 0.
        l = len(vt[vt>=0])
        for i in range(len(v_list)):
            if(list(vt[vt>=0])[i] == list(v[vt>=0])[i] and list(vt[vt>=0])[i] == 1):
                s += 1. 
#                print(i, s)
        s_count.append(s)
#        print(s/l, l)
        temp_s_l.append(s/l)
        temp.append(s)

author1['s'] = temp
author1['s/l'] = temp_s_l

data = {'score': temp}
df = pd.DataFrame(data)
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
temp = list(df_normalized[0])

#result = []
#author1['s/l'] = temp_s_l
#for i in range(len(temp)):
#    if( temp_s_l[i] ==0):
#        result.append(0)
#    else:
#        result.append(temp[i] / temp_s_l[i])
#data = {'score': result}
#df = pd.DataFrame(data)
#
#min_max_scaler = preprocessing.MinMaxScaler()
#np_scaled = min_max_scaler.fit_transform(df)
#df_normalized = pd.DataFrame(np_scaled)
#result = list(df_normalized[0])


author1['No. EV related Matched Keywords'] = s_count
author1['Calculated Result'] = temp




#div = (max(temp) + min(temp))/2.0
div = 0.5
Y_N = []
for i in range(len(temp)):
    if(temp[i] >= div):
        Y_N.append('Y')
    else:
        Y_N.append('N')

my_calculation_comparison = author1[:]
del my_calculation_comparison['Chenwei']
del my_calculation_comparison['1,2,3 gram keywords number']
del my_calculation_comparison['1,2,3 gram No keywords']
del my_calculation_comparison['1,2,3 gram No keywords number']
del my_calculation_comparison['1,2,3 gram']
del my_calculation_comparison['1,2,3 gram numbers']
del my_calculation_comparison['1,2,3 gram keywords']
del my_calculation_comparison['s']
del my_calculation_comparison['s/l']

my_calculation_comparison.to_csv('Result.csv')

my_calculation_comparison['Calculated matching keywords'] = Y_N
author1['Calculated Y/N by 0.5 Threshold'] = Y_N
author1.to_csv('Result.csv')

######################################################################################################
author1_Haley_list = list(my_calculation_comparison.Haley)
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
        author1_Haley_list[i] = 0
    else:
        continue
my_calculation_comparison['Haley'] = author1_Haley_list

author1_Haley_list = list(my_calculation_comparison['Peng Yu'])
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
        author1_Haley_list[i] = 0
    else:
        continue
my_calculation_comparison['Peng Yu'] = author1_Haley_list

author1_Haley_list = list(my_calculation_comparison.Karthik)
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
        author1_Haley_list[i] = 0
    else:
        continue
my_calculation_comparison['Karthik'] = author1_Haley_list

author1_Haley_list = list(my_calculation_comparison['Yi Weng'])
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
        author1_Haley_list[i] = 0
    else:
        continue
my_calculation_comparison['Yi Weng'] = author1_Haley_list

author1_Haley_list = list(my_calculation_comparison['Xiaohuan'])
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
        author1_Haley_list[i] = 0
    else:
        continue
my_calculation_comparison['Xiaohuan'] = author1_Haley_list

author1_Haley_list = list(my_calculation_comparison['Calculated matching keywords'])
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
        author1_Haley_list[i] = 0
    else:
        continue
my_calculation_comparison['Calculated matching keywords'] = author1_Haley_list

my_calculation_comparison.to_csv('my_calculated_comparision_dataframe.csv')

y_pred1 = list(my_calculation_comparison['Calculated matching keywords'])

y_test1 = list(my_calculation_comparison['Haley'])

y_test1 = list(my_calculation_comparison['Peng Yu'])

y_test1 = list(my_calculation_comparison['Karthik'])

y_test1 = list(my_calculation_comparison['Yi Weng'])


# Making the Confusion Matrix
y_pred =[]
y_test =[]
from sklearn.metrics import confusion_matrix
for i in range(len(y_pred1)):
    if(math.isnan(y_test1[i])):
       continue
    else: 
        y_pred.append(y_pred1[i])
        y_test.append(y_test1[i])
cm = confusion_matrix(y_test, y_pred)

accuracy=0
s=0
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if(i==j):
            accuracy += cm[i][j]
        s += cm[i][j]
accuracy = (accuracy/s) * 100

TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]

recall = TP/(TP+FN)
precision = TP/(TP+FP)

print('Precision: '  + str(precision) + '\n' + 'Recall: ' + str(recall)  + '\n' + 'Accuracy: ' + str(accuracy))

author1['Calculated Y/N by 0.5 Threshold'] = Y_N
del author1['Chenwei']
del author1['1,2,3 gram keywords number']
del author1['1,2,3 gram No keywords']
del author1['1,2,3 gram No keywords number']
del author1['1,2,3 gram']
del author1['1,2,3 gram numbers']
del author1['1,2,3 gram keywords']
del author1['s']
del author1['s/l']
author1.to_csv('Result.csv')
