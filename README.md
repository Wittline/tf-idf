# TF-IDF (Term Frequency-Inverse Document Frequency) from Scratch

This technique is a combination of two count-based metrics, Term frequency (tf) and Inverse document frequency (idf), is part of the information retrieval and text feature extraction areas.

Mathematically, **TFIDF** is the product of two metrics:



**Term frequency (tf)**, is the Bag of words model, is denoted by the frequency value of each word in a particualr document and is represented below as.


**Inverse document frequency (idf)** is the inverse of the document frequency for each word, we divide the number of documents by the document frequency for each word, this operation is being scaled using the logarithmic, the formula is adding 1 to the document frequency for each word to highlight that it also has one more document in the corpus, It is also addig 1 to the whole result to avoid ignore terms that could have zero.


**df(word)** represents the number of documents in which the word w is present.

The final TFIDF computed could be normalized dividing the reuslt by L2 normor euclidean norm.












