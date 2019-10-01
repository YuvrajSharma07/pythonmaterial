import nltk
#nltk.download()
import os
import nltk.corpus
'''print(os.listdir(nltk.data.find('corpora')))
from nltk.corpus import brown
print(nltk.corpus.brown.fileids())'''
hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
hamlet
for word in hamlet[:500]:
    print(word, end='_')