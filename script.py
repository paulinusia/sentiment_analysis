
from collections import Counter
from html.parser import HTMLParser
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import math
from sklearn.model_selection import train_test_split
import string
import re
import dictionary

import sys


# EACH VECTOR ID A REVIEW AKA EACH ROW IS A REVIEW
# TAKE PIZZA AND PUT 1 IN THE PIZZA COLUMN IN THE ROW AND DO IT OVER AND OVER AGAIN

'''
Pre-process data
Your task is to build a binary classifier that will perform movie review classifica-tion automatically. 
You will need to implement (i.e. write from scratch) yourown vectorizer to convert the text of the movie reviews into vectors that can beused 
to train a scikit-learn classifier. You will also implement (i.e. write fromscratch) your own model selection procedure. Conretely,
 here are the steps thatyou need to take (each is worth 20 points):

  Randomly split the data into training (80%) and test (20%) sets. Keepthe test set away in a separate location until you are done with modelselection.
 
  2.Implement your own vectorizer that converts the training data into anumpy array of shape (num_of_training_examples x num_of_features).
  To control the dimensionality of the resulting vectors, you can discard thefeatures that occur in fewer thannexamples and/or more thanmexamples.
  You will pick the actual value of n (e.g. 5) and/or m (e.g. 5000) in theprocess of model selection.
   Note: you are not allowed to use third-partyimplementations such as scikit-learn’s 
   CountVectorizer etc.
   
   3.Train a classifier of your choice (e.g. logistic regression, SVM) using n-foldcross validation. 
   You are required to implement your own n-fold cross-validation rather than using the scikit-learn API. I.e. you will need torandomly 
   split the training set intonparts and repeatedly train on n-1parts and test on the remaining part as we discussed in class.
  
  
   4.Implement your own grid search procedure which should include a searchover at least two hyper-parameters one of which could be the 
   parameterthat controls the dimensionality of your data and the other one is aclassifier-specific hyper-parameter such as C in logistic 
   regression andSVM.
   
   5.Use the vectorizer you created in step 2 to convert the test set into a numpyarray of shape: (num_of_test_examples x num_of_features). 
   Pick thebest model and report its performance on the test set. Think about whatperformance metric is appropriate for this data and justify your choice.
'''

def label_data(label, d):
   with open(d) as data:
      for row in data:
         df.loc[len(df)] = [filter_data(row), label]
      return df

def filter_data(data):
   pattern = r'[^a-zA-Z0-9\s]'
   data = data.lower()
   data = re.sub(pattern, '', data)
   data = re.sub(r'\d+', '', data)
   return data


def build_dictionary(d, max, min):
   master = []
   remove = []
   ignore = ['a', "the", "is", "this", "of", "an", "on", "to", "be", "are", "that", "and", "as", "its",
             "in", "it", "with", "for", "you", "has", "at", "not", "will", "or", "i", "their", "can", "way"]
         
   with open(d, 'r') as datafile:
      #print(type(datafile))

      for row in datafile:
         
         data = filter_data(row)
         data = data.split()
         #print(row)
         for word in data:
            if word not in ignore:
               master.append(word)
         
   master = Counter(master)

   for word in master:
      if master.get(word) < min or master.get(word) > max:
            remove.append(word)

   for key in remove:
      del master[key]

   #to ensure that no duplicate values are in the dictionary
   print(sorted(list(set((master.elements())))), file=open("dictionary.py", "a"))
   
   #print(sorted(list(set((master.elements())))))
         #return list(master.elements())


def vectorize(datafile, dict, dataname):
      master = np.zeros((len(datafile), len(dict)))
      master =  pd.DataFrame(data=master, columns=dict)
      #print(master)
      col = 0
      #print(len(datafile['comment']))
      for row in datafile['comment']:
            
            row = row.split()
            for y in row:
               for x in dict:
                  if x == y:
                       master.at[col, x] =  1
                       #print(col, x)
            col = col +1
                  

      #print(master)
            #print(len(new_row), '/', len(dict))

            #print(len(master),' /', len(datafile) )
      
      
      #pd.Dataframe(master, columns=columns) =  df.to_csv('output', sep='\t', encoding='utf-8')
      export_csv = master.to_csv(dataname, index=None, header=True)           
     
def log_reg(train, y_train, test, y_test):
      log = LogisticRegression(penalty='l2', C=1.0, solver='warn')
      log.fit(train, y_train)
      score = log.score(test, y_test)
      return score

def k_fold_log_reg(k_fold_val, data, y_val):
     
      max_iter = [175, 200]
      c_params = [.01, 1.0, 10]

      t = pd.read_csv(data)
      #print(t)
      #print(y_val)
      labeled =  pd.concat([t, y_val], axis=1, sort=False)
      
      #print(labeled)
      #list of dataframes
      folds = np.array_split(labeled, k_fold_val)
      
      #count = k_fold_val -1
      print('     GRID SEARCH    ')
      best_score = round(0,2)
      for x in max_iter:
         for y in c_params:
               average_score = 0
               
               for i in range(k_fold_val-1):
                     
                     training_data = folds.copy()
                     #print(training_data)

                     test_data = folds[i]
                     del training_data[i]
                     training_data = pd.concat(training_data, sort=False)
                     #print(training_data)
                     #print(test_data)
                     #print('## LOG REG:', 'count,' ,'         ##')
                     #count = count - 1     
                     
                     score = perform(training_data, test_data, x, y)
                     average_score = average_score + score
                     
               average_score = round(((average_score/(k_fold_val-1)) *100), 2)
               #print('MAX_ITER: ', x, '     C PARAM: ', y)
               print('ITER:', x, 'C PARAM:', y,
                     '           average_score',  average_score, '%')
               
               if average_score>best_score:
                     best_score = average_score
                     print('MAX_ITER: ', x, '     C PARAM: ', y,'                Best tuned param performance:            ', best_score, '%')    

def perform(train_set, test_set, x, y):
   
   train_labels = train_set['label'].reset_index(drop=True).astype('int')

   train_set = train_set.drop('label', axis=1).reset_index(drop=True).astype('int')


   #train_set = train_set.reset_index(drop=True).astype('int')

   #train_labels = train_labels.reset_index(drop=True).astype('int')

   test_labels = test_set['label'].reset_index(drop=True).astype('int')

   test_set = test_set.drop('label', axis=1).reset_index(drop=True).astype('int')


   #test_set = train_set.reset_index(drop=True).astype('int')

   #test_labels = train_labels.reset_index(drop=True).astype('int')


   #print(type(train_set))
   #print(type(train_labels))
   #print('train set', "\n", train_set)
   #print('test set', "\n", train_labels)



   #print('type of test set:', type(test_set))
   #print('type of test labels', type(test_labels))
   #print(test_set)
   #print(test_labels)
   

   
      
   log = LogisticRegression(penalty='l2', C=y, max_iter=x,solver='lbfgs')
   log.fit(train_set, train_labels)
   score = log.score(test_set, test_labels)
   #print('score: ',round(score* 100,2), '%')
   return score
   
def streamline(data, dict, y_val, folds, datafilename):
       export_csv = data.to_csv(datafilename, index=None, header=True)
       vectorize(data, dict, './vector_test.csv')

       k_fold_log_reg(folds, './vector_test.csv', y_val)

       

if __name__ == '__main__':
      
   columns = ['comment', 'label']
   df  = pd.DataFrame(columns=columns)


   #load, label and shuffle data
   pos_data = label_data(1, './data/rt-polarity.pos')
   data = label_data(0, './data/rt-polarity.neg')
   #print(type(data))
   #print(type(pos_data))
   data.append(pos_data)

   d = shuffle(data, random_state=124)

   #reset index
   data = d.reset_index(drop=True)
   
   # Don't forget to add '.csv' at the end of the path
   #export_csv = data.to_csv(r'data.csv', index=None, header=True)

   ''' SPLIT DATA'''
   train, test = train_test_split( data, test_size=0.2, random_state=1)
   #train, test = (data.iloc[:round(len(data) * .80)])
   #test = pd.concat([train, data])

   #test = test.drop_duplicates(subset="comment")

   y_train = train['label'].reset_index(drop=True)
   train = train.drop('label', axis=1).reset_index(drop=True)

   y_test = test['label'].reset_index(drop=True)

   test = test.drop('label', axis=1).reset_index(drop=True)
   
   #print(train)
   #print(test)
   '''Builds dictionary of all of the words in the positive/negative categories'''



   #build_dictionary('./data.csv', 185, 5)


  

   #print('y_test',y_test)
   #print(test)
   listed = dictionary.dict
   #print('dictionary length ',len(listed))
   
   #export_csv = train.to_csv(r'train.csv', index=None, header=True)

   #vectorize(train, listed )

   #k_fold_log_reg(10,'./vector_train.csv', y_train)
   #k_fold_log_reg(10, './vector_train.csv', y_train)
   streamline(train, listed, y_test, 10, 'train.csv')
   streamline(test, listed, y_test, 10, 'test.csv')







