import pandas as pd

imp_paper_data = pd.read_csv('title_rel_assigned_Haley - title_rel_assigned.csv')

paper_id = []
#Haley = list(imp_paper_data['Haley'])
#Peng_Yu = list(imp_paper_data['Peng Yu'])
#Karthik = list(imp_paper_data['Karthik'])
#Yi_Weng = list(imp_paper_data['Yi Weng'])
###########################################################################################################

author1_Haley_list = list(imp_paper_data.Haley)
Haley = author1_Haley_list

for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
            author1_Haley_list[i] = 0
    else:
        continue

author1_Haley_list = list(imp_paper_data['Peng Yu'])
Peng_Yu = author1_Haley_list

for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
            author1_Haley_list[i] = 0
    else:
        continue

author1_Haley_list = list(imp_paper_data.Karthik)
Karthik = author1_Haley_list
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
            author1_Haley_list[i] = 0
    else:
        continue

author1_Haley_list = list(imp_paper_data['Yi Weng'])
Yi_Weng = author1_Haley_list
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
            author1_Haley_list[i] = 0
    else:
        continue

author1_Haley_list = list(imp_paper_data['Xiaohuan'])
for i in range(len(author1_Haley_list)):
    if(author1_Haley_list[i] == 'Y?' or author1_Haley_list[i] == 'Y*' or author1_Haley_list[i] == 'Y'):
        author1_Haley_list[i] = 1
    elif(author1_Haley_list[i] == 'N?' or author1_Haley_list[i] == 'N*' or author1_Haley_list[i] == 'N'):
            author1_Haley_list[i] = 0
    else:
        continue
######################################################################################################


index2 = []

for i in range(len(Haley)):
    if(Haley[i] == 1):
        if(Haley[i] == Peng_Yu[i] or Haley[i] == Karthik[i] or Haley[i] == Yi_Weng[i]):
            index2.append(i)
        elif(Peng_Yu[i] == Karthik[i] or Peng_Yu[i] == Yi_Weng[i]):
            index2.append(i)
        elif(Karthik[i] == Yi_Weng[i]):
            index2.append(i)
        else:
            continue
    
index3 = []
for i in range(len(Haley)):
    if(Haley[i] == 1):
        if(Haley[i] == Peng_Yu[i] == Karthik[i] or Haley[i] == Karthik[i] == Yi_Weng[i] or Haley[i] == Yi_Weng[i] == Peng_Yu[i] or Peng_Yu[i] == Karthik[i] == Yi_Weng[i]):
            index3.append(i)
  

keyword2_PMID= []
for i in range(len(index2)):
    keyword2_PMID.append(imp_paper_data['PMID'][index2[i]])
    
    
keyword3_PMID= []
for i in range(len(index3)):
    keyword3_PMID.append(imp_paper_data['PMID'][index3[i]])
    
################################################################################################
    
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

# Importing the dataset
data = pd.read_csv('my_calculated_comparision_dataframe.csv', engine = 'python', encoding = 'latin-1')

PMID = data['PMID']

no_keywords = []
keywords = []
key_PMID = keyword2_PMID
for i in range(len(key_PMID)):  
    sock = urllib.request.urlopen("https://www.ncbi.nlm.nih.gov/pubmed/?term=" + str(key_PMID[i])) 
    htmlSource = sock.read()                            
    sock.close()                                        
#    print (htmlSource)  
    
    string = str(htmlSource)
    
    start = string.find('<div class="keywords"><h4>KEYWORDS: </h4><p>')
    if(start == -1):
        no_keywords.append(key_PMID[i])
        print('NO', i)
        continue
    else:
        start = start + len('<div class="keywords"><h4>KEYWORDS: </h4><p>')
        string = string[start:]
        
        stop = string.find('</p></div>')
        stop = stop - len('</p></div>')
        
        string = string[: stop]
        string = string.split(sep = ';')
        for j in range(len(string)):
            string[j] = string[j].lstrip()
            string[j] = string[j].rstrip()
        keywords.append(string)
        print('Yes', i)



keyword_file = open("Keyword_file_imp_paper_2.txt", "w")
keyword_file.write(str(keywords))

new_PMID_list = []
# keyword3_PMID to keyword2_PMID
for i in range(len(key_PMID)):
    if(key_PMID[i] in no_keywords):
        continue
    else:
        new_PMID_list.append(key_PMID[i])    
        
PMID_Keywords = defaultdict(list)
for i in range(len(keywords)):
    PMID_Keywords[new_PMID_list[i]].append(keywords[i])

PMID_Keywords_Dataframe = pd.DataFrame()
PMID_Keywords_Dataframe['PMID'] = PMID_Keywords.keys()
PMID_Keywords_Dataframe['Keywords'] = keywords


no_keywords_file = open("No_Keywords_imp_paper_2.txt", "w")
no_keywords_file.write(str(no_keywords))

csv_file = open("Dataframe_PMID_Keywords_imp_paper_2.csv", "w") 
PMID_Keywords_Dataframe.to_csv(csv_file)
csv_file.close()

d = pd.read_csv('Dataframe_PMID_Keywords_imp_paper_2.csv')

key_words = d['Keywords']

for i in range(len(d)):
    key_words[i] = ast.literal_eval(key_words[i])

keyword_corpus = []    
for i in range(len(key_words)):
    for j in range(len(key_words[i])):
        keyword_corpus.append(key_words[i][j])

txt_keyword_corpus = open("keyword_corpus_imp_paper_2.txt", "w") 
txt_keyword_corpus.write(str(keyword_corpus))
txt_keyword_corpus.close()

keys = open("keyword_corpus_imp_paper_2.txt", "r")
keys = ast.literal_eval(keys.read())
keys = set(keys)
keys.remove('')   
keys = list(keys)
