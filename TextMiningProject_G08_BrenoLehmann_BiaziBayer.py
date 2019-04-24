
# coding: utf-8

# # Text MultiClass Classification - HelpDesk DataSet

# ## Context
# 
# The problem addressed in this project is based on actual data coming from a failure reporting system from a consulting and helpdesk company. Where customers report a failure in a short text message. 
# 
# The service structure follows a three-tier structure:
# * Level 1: Characterized by low technical level, dealing only with basic technical failures.
# * Level 2: Greater technical expertise and transversal characteristics of problem solving.
# * Level 3: team of specialists, who effectively have knowledge to provide solutions to customers.
# 
# 
# The flow of complaints is carried out from Level 1 to Level 3. Since the first levels are usually assigned the function of which level three team should receive the call. Adding to the fact that many customers, to cut costs, do not hire services that are supported at Level 2.
# 
# This call forwarding is routinely done in the wrong way. Thus causing inconsistencies in the flow of treatment and causing loss of efficiency.
# 
# In this way, our project aims to create a tool to help the Level 1 attendants to better indicate which team should treat the call. Consequently, reducing response time and operating costs.
# 
# ## Strategies
# 
# 1 - Data Acquisition
# 
# 2 - Analizing the Dispersion of the Classes
# 
# 3 - Analyze Used Languages
# 
# 4 - Preprocessing Data (Normalize)
# 
#     4.1 - Tokenize the text data
#     4.2 - Normalize The Text
#         4.2.1 - Remove Simbols
#         4.2.2 - Remove Spaces
#         4.2.3 - Remove Stop Words
#         4.2.4 - Steeming the data text 
# 
# 5 - Split Train and Test dataset
# 
# 6 - Train The models
# 
#     5.1 Using as Feature Extraction: Word2Vec, TF-IDF and TF-IDF with oversampling.
#         5.1.1 - Stochastic Gradient Descent
#         5.1.2 - Extra Tree Classifier
#         5.1.3 - Adaboost Classifier
#         5.1.4 - Random Forest Classifier
#         5.1.5 - LogisticRegression
#         5.1.6 - Decision Tree Classifier
#         5.1.7 - Suport Vector Classifier
#         5.1.8 - K-Neighbors Classifier (Only for TF-IDF)
#     
# 6 - Compaire Models
#     
#     6.1 - Use F1-Score and Confusion Matrix as metrics to evaluate the best model
# 
# 7 - Conclusions

# ### 1 - Data Acquisition
# 
# The dataset consists of two fields, with the error messange description and the team that closed the incident.

# In[1]:


import pandas as pd

# Importing the Dataset

### Modify the Path!!! ###
base_dir = r'D:\Mestrado\Text Mining\Project'

df = pd.read_csv(base_dir+'\Incident-Classifier-Dataset.txt', sep = ';', lineterminator='\n')


# In[2]:


# Total Amount of Complains
len(df)


# In[3]:


# Just changing the columns names
df.columns = ['description', 'target']
df['target'] = df['target'].str.rstrip('\r')
df.head()


# In[4]:


print(type(df))
df.head()


# In[5]:


import matplotlib.pyplot as plt


# ----------------------- / / -----------------------
# 
# 
# ### 2 - Analizing the Dispersion of the Classes
# 
# First we check the quality of the targets present in the dataset. In order to avoid inconsistencies and misclassification failures
# 

# In[6]:


# Defining the size of the plot
fig = plt.figure(figsize=(12,6), dpi= 80, facecolor='w', edgecolor='k')
# Ploting the bar plot
df.groupby('target').description.count().plot.bar(ylim=0)


# The dataset has to much diferent classes. But as the plot above shows, there are some input failures, lower case X upper case. 
# 
# Because of that, we must to preprocess the input classes using the expertise and experience of one employe.
# 
# ### Building a Pareto Plot for the original classes distribuition.

# In[7]:


df_pareto = df['target'].value_counts() #generate counts
# Transforming the "Series" into a dataframe
df_pareto = df_pareto.to_frame().reset_index()
# Rename the columns
df_pareto.columns = ['teams', 'n_complains']
# Use the teams column as the index
df_pareto.set_index('teams')


# In[8]:


df_pareto.head()


# In[9]:


# Pareto Plot -> By Classes
from matplotlib.ticker import PercentFormatter

df_pareto["cumpercentage"] = df_pareto["n_complains"].cumsum()/df_pareto["n_complains"].sum()*100


# In[10]:


# Pareto Plot

fig, ax = plt.subplots(figsize=(12,6), dpi= 80, facecolor='w', edgecolor='k')

ax.bar(df_pareto.index, df_pareto["n_complains"], color="C0")
ax2 = ax.twinx()
ax2.plot(df_pareto.index, df_pareto["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()


# In[11]:


df_pareto


# In[12]:


len(df_pareto)


# #### Transforming all Teams Descriptions into UPPER CASE

# In[13]:


df['upper_target'] = df['target'].str.upper()


# In[14]:


df.head()
# df.description.isnull().sum()


# In[15]:


df_pareto = df['upper_target'].value_counts() #generate counts
# Transforming the "Series" into a dataframe
df_pareto = df_pareto.to_frame().reset_index()
# Rename the columns
df_pareto.columns = ['teams', 'n_complains']
# Use the teams column as the index
df_pareto.set_index('teams')
df_pareto.head()


# In[16]:


len(df_pareto)


# In[17]:


# Pareto Plot -> By Classes
from matplotlib.ticker import PercentFormatter

df_pareto["cumpercentage"] = df_pareto["n_complains"].cumsum()/df_pareto["n_complains"].sum()*100


# In[18]:


df_pareto


# We originally had 36 teams receiving the incidents. However, we know that there are some input failures of teams and teams that should not be flagged for fault resolution. Because of this, we will clean the dataset using the technical knowledge of a company employee.

# #### Joining some teams that have changed over time and with multiple descriptions

# In[19]:


# Cleaning and Organazing the Target

# Uniting all "JMS" entries
df.loc[df.upper_target.str.startswith('WPL.JMS'), 'upper_target'] = 'WPL.JMS'
# Uniting all Operation entries
df.loc[df.upper_target.str.contains('OPERA'), 'upper_target'] = 'DCS.OPERATIONS'
# Uniting Storage and Backups with Plataforma
df.loc[df.upper_target.str.contains('STORAGE'), 'upper_target'] = 'DCS.MRS-PLATFORM'
df.loc[df.upper_target.str.contains('BACKUPS'), 'upper_target'] = 'DCS.MRS-PLATFORM'


# In[20]:


df_pareto = df['upper_target'].value_counts() #generate counts
# Transforming the "Series" into a dataframe
df_pareto = df_pareto.to_frame().reset_index()
# Rename the columns
df_pareto.columns = ['teams', 'n_complains']
# Use the teams column as the index
df_pareto.set_index('teams')
df_pareto.head()


# In[21]:


# The amount of Classes after the second round of preprocessing
print(len(df_pareto))


# In[22]:


# df_pareto


# In[23]:


# Pareto Plot -> By Classes
from matplotlib.ticker import PercentFormatter
import numpy as np

total = np.sum(df_pareto.loc[:,'n_complains':].values)
df_pareto['percent'] = df_pareto.loc[:,'n_complains':].sum(axis=1)/total * 100

#df_pareto["percentage"] = df_pareto["n_complains"].sum()/df_pareto["n_complains"].sum()*100
df_pareto["cumpercentage"] = df_pareto["n_complains"].cumsum()/df_pareto["n_complains"].sum()*100


# In[24]:


len(df)


# In[25]:


df_pareto


# In[26]:


#fig = plt.figure(figsize=(12,6), dpi= 80, facecolor='w', edgecolor='k')

fig, ax = plt.subplots(figsize=(12,6), dpi= 80, facecolor='w', edgecolor='k')

ax.bar(df_pareto.index, df_pareto["n_complains"], color="C0")
ax2 = ax.twinx()
ax2.plot(df_pareto.index, df_pareto["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()


# #### Extracting The Minority Classes Based on the Pareto Plot
# 
# Based on the result of the pareto plot, we decided to aply as a cut off 98% of entries. 

# We decided to remove the classes listed in the variable cuf off, since they refer to incidents closed by teams that do not have the technical competence to close. Therefore, it is an input failure.

# In[27]:


cut_off = ['PROBLEM_MANAGEMENT','WPL.3RDPARTY','WPL.ORDERS','DCS.SHAREPOINT',
           'WPL.AVARIAS','DCS.CLOUD-AUTOMATION','DCS.INFRA-SETUP','DCS.TOOLS',
           'DCS.REPORTING','NM.NETWORKING','DCS.VIRTUALIZACAO','DCS.OPERATIONS',
           'DCS.CCL','WPL.HELPDESK','DCS.SERVICE-MANAGER']


# In[28]:


for i in cut_off:
    df = df[df.upper_target != i]


# In[29]:


len(df)


# In[30]:


df_pareto = df['upper_target'].value_counts() #generate counts
# Transforming the "Series" into a dataframe
df_pareto = df_pareto.to_frame().reset_index()
# Rename the columns
df_pareto.columns = ['teams', 'n_complains']
# Use the teams column as the index
df_pareto.set_index('teams')
print()


# In[31]:


# Pareto Plot -> By Classes
from matplotlib.ticker import PercentFormatter
import numpy as np

total = np.sum(df_pareto.loc[:,'n_complains':].values)
df_pareto['percent'] = df_pareto.loc[:,'n_complains':].sum(axis=1)/total * 100

#df_pareto["percentage"] = df_pareto["n_complains"].sum()/df_pareto["n_complains"].sum()*100
df_pareto["cumpercentage"] = df_pareto["n_complains"].cumsum()/df_pareto["n_complains"].sum()*100


# In[32]:


df_pareto


# In[33]:


#fig = plt.figure(figsize=(12,6), dpi= 80, facecolor='w', edgecolor='k')

fig, ax = plt.subplots(figsize=(12,6), dpi= 80, facecolor='w', edgecolor='k')

ax.bar(df_pareto.index, df_pareto["n_complains"], color="C0")
ax2 = ax.twinx()
ax2.plot(df_pareto.index, df_pareto["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.show()


# In[34]:


len(df)


# ###### After excluding the minority classes we have 69360 calls the support teams. Distributed among 9 classes.

# In[35]:


# Saving this first round of data preparation into a text file
df.to_csv(base_dir+'\Incident-Classifier-Dataset_V2.txt', sep = ';')


# In[36]:


df.head()


# ----------------------- / / -----------------------
# 
# 
# 
# # 3 - Analyze Used Languages
# 
# All calls to the technical team may not necessarily always be described in the same language. In this way, we have done an exploration of the data in order to know what is the dispersion of the languages ​​used in the description of the failures.

# In[37]:


from langdetect import detect


# In[38]:


# Commented code, since it is only used to create a check point and 
#speed up the most future tests in the code.

#language = pd.read_csv(base_dir+'\language.txt', sep = ';', lineterminator='\n')
#language.columns = ['false_index','language']
#language = language['language']
#language = language.str.rstrip('\r')


# First we exclude as entries that have empty descriptions

# In[39]:


df = df[pd.notnull(df['description'])]
print(len(df))


# In[40]:


language_description = df['description']


# In[41]:


df.head()


# In[42]:


get_ipython().run_cell_magic('time', '', "lang = []\n\nfor i in language_description:\n    # capitalize the item and add it to regimentNamesCapitalized_f\n    lang.append(detect(i))\n\nlanguage = pd.DataFrame(lang, columns = list('a'))")


# In[43]:


from collections import Counter
#Counter(lang)


# In[44]:


#print(len(lang))
print(len(language))
print(len(df))


# In[45]:


df = df.reset_index()


# In[46]:


df.drop(columns=['index'])


# In[47]:


df = pd.concat([df, language], axis=1)


# In[48]:


len(df)


# In[49]:


df = df.drop(columns=['index'])


# In[50]:


df.columns = ['description','target','new_target','language']
df['qtd'] = 1


# In[51]:


df.head()


# In[52]:


len(df)


# Saving this dataset as the result of an intermediate processing as a text document.

# In[53]:


# language.to_csv(base_dir+'\language.txt', sep = ';')
# df.to_csv(base_dir+'\Incident-Classifier-Dataset_V3.txt', sep = ';')


# #### Showing the distribuition of languages in the dataset

# In[54]:


from matplotlib.pyplot import pie, axis, show

fig, ax = plt.subplots(figsize=(20,10), dpi= 80, facecolor='w', edgecolor='k')

import matplotlib as mpl
mpl.rcParams['font.size'] = 15.0

sums = df.qtd.groupby(df.language).sum()
axis('equal');
pie(sums, autopct='%1.1f%%', labels=sums.index,textprops={'fontsize': 22});
show()


# In[55]:


sums


# For simplify the model, we choosed to use only the english descriptions. Because with only this language represent 92,3% of all complains.

# In[56]:


# Detect Language 
# df_en = df.loc[df['language'] == 'en']

df_en = df.loc[df['language'].isin(['en'])]

print(len(df_en))


# In[57]:


df_en = df_en[pd.notnull(df['description'])]
print(len(df_en))


# In[58]:


len(df_en)


# In[59]:


df_en.head()


# Now, after all the rounds of data cleaning, we have a dataset with approximately 64k incidents.

# ----------------------- / / -----------------------
# 
# 
# # 4 - Preprocessing Text Description Data
# 
# First, we pre-process all incidents in order to: remove any symbols, remove all numbers, delete all spaces, delete all stopwords and finally steeming the reaming dataset.
# 
# ### 4.1 - Tokenize the text data

# In[60]:


get_ipython().run_cell_magic('time', '', 'from nltk.tokenize import word_tokenize')


# In[61]:


get_ipython().run_cell_magic('time', '', "df_en['tokenized_description'] = df_en['description'].apply(word_tokenize) ")


# In[62]:


type(df_en['tokenized_description'])


# Saving the target into a Text File

# In[63]:


df_en.new_target.to_csv(base_dir+r'\target.txt', sep = ';')


# ### 4.2 - Normalize The Text

# In[64]:


df_en.head()


# Defining the function to normalize the text data.

# In[65]:


import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer


# In[66]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


# In[67]:


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
#     text = text.decode('utf8')
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub(r'\d+', '', text)
    ps = PorterStemmer() # Stemming the Text
    text = ps.stem(text) # Stemming the Text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    
    return text


# In[69]:


get_ipython().run_cell_magic('time', '', "df_en['description_processed'] = df_en.description.apply(lambda x: clean_text(x))")


# In[70]:


df_en = df_en[['description_processed', 'description', 'language', 'qtd', 'target', 'new_target']]


# In[71]:


df_en.head()


# In[72]:


# number of words after the process to normalize the data
df_en['description_processed'].apply(lambda x: len(x.split(' '))).sum()


# In[73]:


# Subset the dataset with only the normalized text data and the processed target
df_en_clean = df_en[['description_processed','new_target']]


# In[74]:


# Saving the intermediate delivery into a text file
df_en_clean.to_csv(base_dir+'\df_en_clean.txt', sep = ';')


# In[75]:


df_en_clean.head()


# # 5 -  Sampling Trainning and Test Set

# In[76]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# In[77]:


x = df_en.description_processed
y = df_en.new_target


# ## Oversapling the Unbaleced DataSet

# In[78]:


from imblearn.over_sampling import RandomOverSampler


# ----------------------------- / / -------------------------------------

# In[83]:


my_tags = df_en.new_target.unique()
my_tags


# -------------------- / / ---------------------
# 
# 
# 
# # 6 - Training the Models
# 
# 
# Our strategy consisted in including using two techniques for feature extraction: Word2Vec and TF-IDF. And especially for TF-IDF we will implement oversample in the trainning set so that the model is not lost in the accuracy paradox.

# ### Confusion Matrix Function

# In[84]:


from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# y_true = encoder.fit_transform(y_test)
# y_pred = encoder.fit_transform(pred)
# len(y)


# In[3]:


# Confusion Matrix Function - Copy from the website below
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

classes = my_tags

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
#     fig.plt.figure(figsize=(30,20))
    fig.tight_layout()
    return ax


# ## Models to Be Tested with the Data
# 
# After the text of the descriptions are processed and converted into a matrix TF-IDF, we have the following predictive models:
# 
# ##### 1 - Suport Vector Classification
# 
# ##### 2 - MultinomialNaiveBayes
# 
# ##### 3 - DecisionTreeClassifier
# 
# ##### 4 - LogisticRegression
# 
# ##### 5 - RandomForestClassifier
# 
# ##### 6 - AdaboostClassifier
# 
# ##### 7 - BaggingClassifier
# 
# ##### 8 - ExtraTreeClassifier
# 
# ##### 9 - Stochastic Gradient Descent Classifier

# In[86]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import _pickle as pkl
import warnings
import csv
from nltk.tokenize import word_tokenize
import sqlite3

import os
from sklearn.metrics import classification_report

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from sklearn.preprocessing import LabelEncoder


# In[87]:


# SuportVectorClassification
svc = SVC(kernel='sigmoid', gamma=1.0)
# KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=49)
# MultinomialNaiveBayes
mnb = MultinomialNB(alpha=0.2)
# DecisionTreeClassifier
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=42)
# LogisticRegression
lrc = LogisticRegression(solver='liblinear', penalty='l1')
# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=31, random_state=42)
# AdaboostClassifier
abc = AdaBoostClassifier(n_estimators=62, random_state=42)
# BaggingClassifier
bc = BaggingClassifier(n_estimators=9, random_state=42)
# ExtraTreeClassifier
etc = ExtraTreesClassifier(n_estimators=9, random_state=42)
# StochasticGradientDescent
sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None) # Stocastic Gradient Descent Classifier
# 


# ### Doc2vec and Logistic Regression
# 
# Word2Vec produce a word embedding, using a two-layer neural network that are trained to reconstruct linguistic contexts of words. Word2Vec plot all words into a vector space, producing word vectors such that words that share common contexts in the corpus are located in close proximity to one another in the space.

# In[88]:


from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re

from sklearn.linear_model import LogisticRegression


# In[89]:


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled


# In[90]:


X_train, X_test, Y_train, Y_test = train_test_split(df_en.description_processed, df_en.new_target, random_state=0, test_size=0.3)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test


# In[91]:


all_data[:2]


# In[93]:


get_ipython().run_cell_magic('time', '', 'model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)\nmodel_dbow.build_vocab([x for x in tqdm(all_data)])\n\nfor epoch in range(3):\n    model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)\n    model_dbow.alpha -= 0.002\n    model_dbow.min_alpha = model_dbow.alpha')


# In[94]:


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors
    
train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')


# In[95]:


clfsW = {'StochasticGradientDescent_Word2Vec': sgd,'ExtraTreeClassifier_Word2Vec': etc, 
         'BaggingClassifier_Word2Vec': bc, 'AdaboostClassifier_Word2Vec': abc, 
        'RandomForestClassifier_Word2Vec': rfc, 'LogisticRegression_Word2Vec': lrc, 
         'DecisionTreeClassifier_Word2Vec': dtc,'SuportVectorClassification_Word2Vec': svc}


# In[96]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[97]:


pred_scores = []
pred_scores


# In[98]:


get_ipython().run_cell_magic('time', '', 'def train_classifier(clf, feature_train, labels_train):   \n    clf.fit(feature_train, labels_train)\n   \n\ndef predict_labels(clf, features):\n    return (clf.predict(features))')


# In[99]:


get_ipython().run_cell_magic('time', '', "for k,v in clfsW.items():\n#     print(k)\n#     print(v)\n    print( 'Training classifier: {}'.format(k))\n    train_classifier(v, train_vectors_dbow, Y_train)\n    pred = predict_labels(v,test_vectors_dbow)\n    # Incluir o Recal, precision and F1 Score\n#     pred_scores.append((k, [accuracy_score(y_test,pred)]))\n\n    pred_scores.append((k, [f1_score(Y_test,pred,average='weighted')]))\n    clfpkl = open( 'output_model_'+k+'.pkl' , 'wb')\n    pkl.dump(v, clfpkl)\n    clfpkl.close()\n    print(pred_scores)\n    print(classification_report(Y_test, pred))\n    \n    encoder = LabelEncoder()\n    y_true = encoder.fit_transform(Y_test)\n    y_pred = encoder.fit_transform(pred)\n    plot_confusion_matrix(y_true, y_pred, classes = classes, normalize=True,\n                      title='Normalized confusion matrix' + ' - ' + format(k))\n    plt.show()")


# In[100]:


# pred_scores.append((k, [f1_score(y_test,pred,average='weighted')]))
print(pred_scores)


# ## Running Models in a TF-IDF Matrix
# 
# Now we will apply again the same models in a TF-IDF matrix

# In[101]:


vectorizer = TfidfVectorizer()
# vectorizer_ = CountVectorizer()


# In[102]:


get_ipython().run_cell_magic('time', '', 'features = vectorizer.fit_transform(x)\n# features_ = vectorizer_.transform(x)')


# In[103]:


# vectorizer_pkl = open( 'vectorizer.pkl' , 'wb')
# pkl.dump(vectorizer,vectorizer_pkl)
# vectorizer_pkl.close()


# In[104]:


# features_train, features_test, labels_train, labels_test = train_test_split(features, df_en_clean['new_target'], test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state = 42)


# Listing all the models to be tested

# In[105]:


# clfs = {'StochasticGradientDescent': sgd,'ExtraTreeClassifier': etc, 'BaggingClassifier': bc, 'AdaboostClassifier': abc, 
#         'RandomForestClassifier': rfc, 'LogisticRegression': lrc, 'DecisionTreeClassifier': dtc, 'MultinomialNaiveBayes': mnb,
#          'SuportVectorClassification': svc, 'KNeighborsClassifier': knc}

clfs = {'StochasticGradientDescent_TfIdf': sgd,'ExtraTreeClassifier_TfIdf': etc, 
        'BaggingClassifier_TfIdf': bc, 'AdaboostClassifier_TfIdf': abc, 
        'RandomForestClassifier_TfIdf': rfc,'LogisticRegression_TfIdf': lrc, 
        'DecisionTreeClassifier_TfIdf': dtc,'MultinomialNaiveBayes_TfIdf': mnb,
        'SuportVectorClassification_TfIdf': svc}

# clfs = {'KNeighborsClassifier': knc}


# In[106]:


get_ipython().run_cell_magic('time', '', "\nfor k,v in clfs.items():\n    print( 'Training classifier: {}'.format(k))\n    train_classifier(v, x_train, y_train)\n    pred = predict_labels(v,x_test)\n    # Incluir o Recal, precision and F1 Score\n#     pred_scores.append((k, [accuracy_score(y_test,pred)]))\n    pred_scores.append((k, [f1_score(y_test,pred,average='weighted')]))\n    clfpkl = open( 'output_model_'+k+'.pkl' , 'wb')\n    pkl.dump(v, clfpkl)\n    clfpkl.close()\n    print(pred_scores)\n    print(classification_report(y_test, pred))\n    \n    encoder = LabelEncoder()\n    y_true = encoder.fit_transform(y_test)\n    y_pred = encoder.fit_transform(pred)\n    plot_confusion_matrix(y_true, y_pred, classes = classes, normalize=True,\n                      title='Normalized confusion matrix')\n    plt.show()")


# ### Testing the Models With Oversampling Train Data

# We make some tests nos with Oversampling train set. Assuming the preliminary results Testing the 

# In[107]:


from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_ROS, y_ROS = ros.fit_sample(x_train, y_train)


# In[108]:


X_ROS.getnnz()


# In[109]:


# clfs_tf_over = {'StochasticGradientDescent_OverSamp_TfIdf': sgd,'ExtraTreeClassifier_OverSamp_TfIdf': etc, 'BaggingClassifier_OverSamp_TfIdf': bc, 
#         'AdaboostClassifier_OverSamp_TfIdf': abc, 'RandomForestClassifier_OverSamp_TfIdf': rfc, 'LogisticRegression_OverSamp_TfIdf': lrc, 
#         'DecisionTreeClassifier_OverSamp_TfIdf': dtc, 'MultinomialNaiveBayes_OverSamp_TfIdf': mnb,
#           'SuportVectorClassification_OverSamp_TfIdf': svc, 'KNeighborsClassifier_OverSamp_TfIdf': knc}

clfs_tf_over = {'StochasticGradientDescent_OverSamp_TfIdf': sgd,'ExtraTreeClassifier_OverSamp_TfIdf': etc, 
                'BaggingClassifier_OverSamp_TfIdf': bc, 'AdaboostClassifier_OverSamp_TfIdf': abc, 
                'RandomForestClassifier_OverSamp_TfIdf': rfc, 'LogisticRegression_OverSamp_TfIdf': lrc, 
                'DecisionTreeClassifier_OverSamp_TfIdf': dtc, 'MultinomialNaiveBayes_OverSamp_TfIdf': mnb,
                'SuportVectorClassification_OverSamp_TfIdf': svc}


# In[110]:


get_ipython().run_cell_magic('time', '', "\nfor k,v in clfs_tf_over.items():\n    print( 'Training classifier: {}'.format(k))\n    train_classifier(v, X_ROS, y_ROS)\n    pred = predict_labels(v,x_test)\n    # Incluir o Recal, precision and F1 Score\n#     pred_scores.append((k, [accuracy_score(y_test,pred)]))\n    pred_scores.append((k, [f1_score(y_test,pred,average='weighted')]))\n    clfpkl = open( 'output_model_'+k+'.pkl' , 'wb')\n    pkl.dump(v, clfpkl)\n    clfpkl.close()\n    print(pred_scores)\n    print(classification_report(y_test, pred))\n    \n    encoder = LabelEncoder()\n    y_true = encoder.fit_transform(y_test)\n    y_pred = encoder.fit_transform(pred)\n    plot_confusion_matrix(y_true, y_pred, classes = classes, normalize=True,\n                      title='Normalized confusion matrix')\n    plt.show()")


# # 6 - Compaire Models
# 
# ### Listing the F1 Scores Of all Models
# 
# We chose to use F1 Score as the primary metric for evaluating the results of the models. Because this is a problem of Multi Class Classification and with an unbalanced dataset. 

# In[111]:


df_score = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
print(df_score)


# The models applied in data preprocessed with Word2Vec, on average, obtained worse results than the models that were based on the TF-IDF. And to prevent the paradox of acuracy. We apply an Oversampling technique in the data set to train the models, thus avoiding the training of the models in an unbalanced dataset.
# 
# And this strategy results in much more stable models in terms of classification of several intended classes. As it shows the a Confusion Matrix below, referring to the chosen model: TF-IDF Logistic Regression in a oversample train set.

# #### ![image.png](attachment:image.png)

# # 7 - Conclusions

# The chosen model, TF-IDF Logistic Regression in an oversample train set, correctly classifies each class approximately 80 to 94% of all classes. In this way, already being able to be applied as a tool usable by the company.
# 

# # 8 - Future Work

# * Testing data in a neural network, at the beginning of our work, we test the dataset in a simple neural network with only one hiden layear. However, the results were extremely disappointing. But, we are aware that to apply the Neural Network, we should have better performed the pre-processing of the data, so as to better prevent overfiting.
# * Test Glove as feature selection.
# * Apply ensemble in the best models.

# # 9 - bibliography
# 
# https://www.onely.com/blog/what-is-tf-idf/
# https://scikit-learn.org/stable/modules/preprocessing.html
# https://machinelearningmastery.com/
# 
# 
# 
