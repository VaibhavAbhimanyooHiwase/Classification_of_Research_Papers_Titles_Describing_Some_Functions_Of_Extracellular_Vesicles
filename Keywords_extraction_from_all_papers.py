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

# Importing the dataset
data = pd.read_csv('title_rel_assigned_Haley - title_rel_assigned.csv', engine = 'python', encoding = 'latin-1')

PMID = data['PMID']
PMID_list = list(PMID)

no_keywords = []
keywords = []
for i in range(len(PMID_list)):  
    sock = urllib.request.urlopen("https://www.ncbi.nlm.nih.gov/pubmed/?term=" + str(PMID_list[i])) 
    htmlSource = sock.read()                            
    sock.close()                                        
#    print (htmlSource)  
    
    string = str(htmlSource)
    
    start = string.find('<div class="keywords"><h4>KEYWORDS: </h4><p>')
    if(start == -1):
        no_keywords.append(PMID_list[i])
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



keyword_file = open("Keyword_file.txt", "w")
keyword_file.write(str(keywords))

new_PMID_list = []
for i in range(len(PMID_list)):
    if(PMID_list[i] in no_keywords):
        continue
    else:
        new_PMID_list.append(PMID_list[i])    
        
PMID_Keywords = defaultdict(list)
for i in range(len(keywords)):
    PMID_Keywords[new_PMID_list[i]].append(keywords[i])

PMID_Keywords_Dataframe = pd.DataFrame()
PMID_Keywords_Dataframe['PMID'] = PMID_Keywords.keys()
PMID_Keywords_Dataframe['Keywords'] = keywords


no_keywords_file = open("No_Keywords.txt", "w")
no_keywords_file.write(str(no_keywords))

csv_file = open("Dataframe_PMID_Keywords.csv", "w") 
PMID_Keywords_Dataframe.to_csv(csv_file)
csv_file.close()

d = pd.read_csv('Dataframe_PMID_Keywords.csv')

key_words = d['Keywords']

for i in range(len(d)):
    key_words[i] = ast.literal_eval(key_words[i])

keyword_corpus = []    
for i in range(len(key_words)):
    for j in range(len(key_words[i])):
        keyword_corpus.append(key_words[i][j])

txt_keyword_corpus = open("keyword_corpus.txt", "w") 
txt_keyword_corpus.write(str(keyword_corpus))
txt_keyword_corpus.close()

keys = open("keyword_corpus.txt", "r")
keys = ast.literal_eval(keys.read())
keys = set(keys)
keys.remove('')   
keys = list(keys)

# Now keys will have all keywords from all papers of PMID