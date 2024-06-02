import praw
import spacy 
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import pandas as pd
import matplotlib.pyplot as plt
import nltk 
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
import networkx as nx
import nltk
import numpy as np

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

reddit = praw.Reddit(client_id= 'KLoTxbEEbEHFF4N0U-E2vg',client_secret = 'yQr2EL3WQ0hqAgB29M03yUrfMxFpXA',user_agent = 'WebScrapping')
lonely_posts =  reddit.subreddit('lonely').hot(limit=1000)
sad_posts = reddit.subreddit('sad').hot(limit=1000)
happy_posts = reddit.subreddit('happy').hot(limit=1000)
disgust_posts = reddit.subreddit('disgusting').hot(limit=1000)
politics_posts = reddit.subreddit('politics').hot(limit=1000)


def remove_stringpreprocessing(str):

    cachedstopwords = stopwords.words("english")
    text = ''
    text = ''.join([word for word in str.split() if word not in cachedstopwords])
    res = ''
    res = ''.join([i for i in text if not i.isdigit()])
    res.replace(" ","")
    return res
    
        

# Because Entities are 2D array
def get_numericalsysnet_entities(temp_list):
    
    

    sysnet0 = list(swn.senti_synsets(remove_stringpreprocessing(temp_list[0])))
    if(len(sysnet0)==0):
        score0 = float(1)
    else:

        score0 = max(sysnet0[0]._pos_score,sysnet0[0]._neg_score,sysnet0[0]._obj_score)
    
    sysnet1 = list(swn.senti_synsets(remove_stringpreprocessing(temp_list[1])))
    if(len(sysnet1)==0):
        score1 = float(1)
    else:
        score1 = max(sysnet1[0]._pos_score,sysnet1[0]._neg_score,sysnet1[0]._obj_score)
    add_list = []
    add_list.append(score0)
    add_list.append(score1)
    
    return add_list
            
             
# Relations are 1-D array
def get_numericalsysnet_relations(str):
    return_list = []
    
    sysnet = list(swn.senti_synsets(remove_stringpreprocessing(str)))
    if(len(sysnet)==0):
        score = 0
    else:

    
        
        score = max(sysnet[0]._pos_score,sysnet[0]._neg_score,sysnet[0]._obj_score)
    
    return score


# extract the subject/object along with its modifiers, compound words and also extract the punctuation marks between them.
def get_entities(sentence):

  
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""   
    prv_tok_text = ""   

    prefix = ""
    modifier = ""

  
    for tok in nlp(sentence):
    
        if tok.dep_ != "punct":
      
            if tok.dep_ == "compound":
                prefix = tok.text
        
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " "+ tok.text
      
      
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
        
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " "+ tok.text
      
       
            if tok.dep_.find("subj") == True:
                ent1 = modifier +" "+ prefix + " "+ tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""      

      
            if tok.dep_.find("obj") == True:
                ent2 = modifier +" "+ prefix +" "+ tok.text
        
   
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
  

    return [ent1.strip(), ent2.strip()]

def get_relation(sentence):
    doc = nlp(sentence)

  
    matcher = Matcher(nlp.vocab)

   
    pattern = [[{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}]]

    matcher.add("matching_1",  pattern) 

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)


# X is float value
def sigmoid(x):
    z = 1/(1+ np.exp(-x))
    return z

def get_value(n1, n2, n3):
    value = 0
    value1 = 0
    value2 = 0
    value3 = 0
    for key in n1.keys():
        value1 = value1 + n1[key]*get_numericalsysnet_relations(key)
    for key in n2.keys():
        value2 = value2 + n2[key]*get_numericalsysnet_relations(key)
    for key in n3.keys():
        value3 = value3 + n3[key]*get_numericalsysnet_relations(key)
    value = sigmoid(value1 + value2 +value3)
    return value



lonely_list = []
for ele in lonely_posts:
    
    lonely_list.append(ele.title)


print(len(lonely_list))

entity_pair_lonelyposts = []
relations_lonelyposts = []
for sentences in lonely_list:
    entity_pair_lonelyposts.append(get_entities(sentence=sentences))
    
    relations_lonelyposts.append(get_relation(sentence=sentences))

source = [i[0] for i in entity_pair_lonelyposts]
target = [i[1] for i in entity_pair_lonelyposts]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations_lonelyposts})
G_lonelypost_digraph=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.DiGraph())



plt.figure(figsize=(12,12))

pos = nx.spring_layout(G_lonelypost_digraph)
nx.draw(G_lonelypost_digraph, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


eigenvector_centrality=nx.eigenvector_centrality(G_lonelypost_digraph)
katz_centrality = nx.katz_centrality(G_lonelypost_digraph)
harmonic_centrality = nx.closeness_centrality(G_lonelypost_digraph)

value_lonelypost = get_value(eigenvector_centrality,katz_centrality,harmonic_centrality)




happy_list = []
for ele in happy_posts:
    
    happy_list.append(ele.title)

print(len(happy_list))

entity_pair_happyposts = []
relations_happyposts = []
for sentences in happy_list:
    entity_pair_happyposts.append(get_entities(sentence=sentences))
    
    relations_happyposts.append( get_relation(sentence=sentences))


source = [i[0] for i in entity_pair_happyposts]
target = [i[1] for i in entity_pair_happyposts]
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations_happyposts})
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.DiGraph())

plt.figure(figsize=(12,12))
plt.show()

eigenvector_centrality=nx.eigenvector_centrality(G)
katz_centrality = nx.katz_centrality(G)
harmonic_centrality = nx.closeness_centrality(G)

value_happypost = get_value(eigenvector_centrality,katz_centrality,harmonic_centrality)
    
    


disgust_list = []
for ele in disgust_posts:
    
    disgust_list.append(ele.title)

print(len(disgust_list))


entity_pair_disgustposts = []
relations_disgustposts = []
for sentences in disgust_list:
    entity_pair_disgustposts.append( get_entities(sentence=sentences))
    
    relations_disgustposts.append( get_relation(sentence=sentences))


source = [i[0] for i in entity_pair_disgustposts]
target = [i[1] for i in entity_pair_disgustposts]
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations_disgustposts})
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using= nx.DiGraph())

plt.figure(figsize=(12,12))



pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


eigenvector_centrality=nx.eigenvector_centrality(G)
katz_centrality = nx.katz_centrality(G)
harmonic_centrality = nx.closeness_centrality(G)

value_disgustpost = get_value(eigenvector_centrality,katz_centrality,harmonic_centrality)

politics_list = []
for ele in politics_posts:
    
    politics_list.append(ele.title)

print(len(politics_list))
entity_pair_politicsposts = []
relations_politicsposts = []
for sentences in politics_list:
    entity_pair_politicsposts.append( get_entities(sentence=sentences))
    
    relations_politicsposts.append( get_relation(sentence=sentences))


source = [i[0] for i in entity_pair_politicsposts]
target = [i[1] for i in entity_pair_politicsposts]
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations_politicsposts})
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using= nx.DiGraph())

plt.figure(figsize=(12,12))



pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


eigenvector_centrality=nx.eigenvector_centrality(G)
katz_centrality = nx.katz_centrality(G)
harmonic_centrality = nx.closeness_centrality(G)

value_politicspost = get_value(eigenvector_centrality,katz_centrality,harmonic_centrality)


sad_list = []
for ele in sad_posts:
    
    sad_list.append(ele.title)
print(len(sad_list))
entity_pair_sadposts = []
relations_sadposts = []
for sentences in sad_list:
    entity_pair_sadposts.append(get_entities(sentence=sentences))
   
    relations_sadposts.append(get_relation(sentence=sentences))

source = [i[0] for i in entity_pair_sadposts]
target = [i[1] for i in entity_pair_sadposts]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations_sadposts})
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using= nx.DiGraph())

plt.figure(figsize=(12,12))



pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


eigenvector_centrality=nx.eigenvector_centrality(G)
katz_centrality = nx.katz_centrality(G)
harmonic_centrality = nx.closeness_centrality(G)

value_sadpost = get_value(eigenvector_centrality,katz_centrality,harmonic_centrality)







