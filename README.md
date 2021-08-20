# TF-IDF (Term Frequency-Inverse Document Frequency) from Scratch

This technique is a combination of two count-based metrics, Term frequency (tf) and Inverse document frequency (idf), is part of the information retrieval and text feature extraction areas, 

Mathematically, **TFIDF** is the product of two metrics, and the final TFIDF computed could be normalized dividing the reuslt by L2 normor euclidean norm.

![image](https://user-images.githubusercontent.com/8701464/130260664-fb928e4c-241f-4d3a-a0f4-8db4dc6ffcf1.png)

**Term frequency (tf)**, is the Bag of words model, is denoted by the frequency value of each word in a particualr document and is represented below as.

![image](https://user-images.githubusercontent.com/8701464/130260732-31e928d5-0c4a-4915-a671-1b4564783c58.png)


**Inverse document frequency (idf)** is the inverse of the document frequency for each word, we divide the number of documents by the document frequency for each word, this operation is being scaled using the logarithmic, the formula is adding 1 to the document frequency for each word to highlight that it also has one more document in the corpus, It is also addig 1 to the whole result to avoid ignore terms that could have zero.


**df(word)** represents the number of documents in which the word w is present.

![image](https://user-images.githubusercontent.com/8701464/130260766-f5734ce8-6981-49d3-861b-97eec6c6a559.png)



The worflow below is showing the steps involved in the compututation of the TFIDF metric:

1. At first, we have to preprocess the text, removing stowwords and special characters.
```
corpus = [
"Love is like pi – natural, irrational, and very important.",
"Love is being stupid together.",
"Love is sharing your popcorn.",
"Love is like Heaven, but it can hurt like Hell."
]

obj = TFIDF(corpus)
obj.preprocessing_text()
```

2. Calculate the frequency of each word for each document (tf)

```
tf = obj.tf()

```

![image](https://user-images.githubusercontent.com/8701464/130262801-3a839159-2d05-4b26-bf41-9930f2388d33.png)


3. Calculate the number of documents in which the word **w** appear

```
df = obj.df(tf)
```

![image](https://user-images.githubusercontent.com/8701464/130262837-9e010d87-3430-4150-880b-810dae4412bf.png)


4. Idf must be calculated using the formula describes above

```
idf, idf_d = obj.idf(df)
```

![image](https://user-images.githubusercontent.com/8701464/130262859-a8750042-112b-47b1-8bb1-44f96ebf60da.png)


5. TFIDF needs the two metric already calculated, TF and IDF, the final results is being normalized using L2 norm

```
tfidf = obj.tfidf(tf, idf)
```

```
df = pd.DataFrame(np.round(tfidf,2), columns= list(tf.columns))
sorted_column_df = df.sort_index(axis=1)
sorted_column_df
```
![image](https://user-images.githubusercontent.com/8701464/130262893-37e230a7-a0a0-41a0-b520-6f6c064e3e76.png)


![image](https://user-images.githubusercontent.com/8701464/130260824-f8c8eef3-4256-4355-9276-71503c7b16bb.png)


# CODE

```
import pandas as pd
import numpy as np
import re
import nltk
from collections import Counter
import scipy.sparse as sp
from numpy.linalg import norm

class TFIDF(object):

    def __init__(self, corpus):        
        self.corpus = corpus
        self.norm_corpus  = None        

    def __normalize_corpus(self, d):
        stop_words = nltk.corpus.stopwords.words('english')
        d = re.sub(r'[^a-zA-Z0-9\s]', '', d, re.I|re.A)
        d = d.lower().strip()
        tks = nltk.word_tokenize(d)
        f_tks = [t for t in tks if t not in stop_words]
        return ' '.join(f_tks)

    def preprocessing_text(self):
        n_c = np.vectorize(self.__normalize_corpus)
        self.norm_corpus = n_c(self.corpus)

    def tf(self):
        words_array = [doc.split() for doc in self.norm_corpus]
        words = list(set([word for words in words_array for word in words]))
        features_dict = {w:0 for w in words}
        tf = []
        for doc in self.norm_corpus:
            bowf_doc = Counter(doc.split())
            all_f = Counter(features_dict)
            bowf_doc.update(all_f)
            tf.append(bowf_doc)
        return pd.DataFrame(tf)

    def df(self, tf):
        features_names = list(tf.columns)
        df = np.diff(sp.csc_matrix(tf, copy=True).indptr)
        df = 1 + df
        return df
        
    def idf(self, df):
        N = 1 + len(self.norm_corpus)
        idf = (1.0 + np.log(float(N) / df)) 
        idf_d = sp.spdiags(idf, diags= 0, m=len(df), n= len(df)).todense()      
        return idf, idf_d

    def tfidf(self, tf, idf):        
        tf = np.array(tf, dtype='float64')
        tfidf = tf * idf
        norms = norm(tfidf , axis=1)
        return (tfidf / norms[:,None])
```















