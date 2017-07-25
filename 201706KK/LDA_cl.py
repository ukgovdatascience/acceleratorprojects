
# coding: utf-8

from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim
import csv
import pandas as pd
import nltk
import re
from re import sub
import sys

"""
Creates LDAVis from command line.

Expects two arguments:

    * input csv file (containing corpus)
    * output file name (Html visualisation)
    * number of topics

"""

input_file = sys.argv[1]
vis_file_name = sys.argv[2]

if not None:
    num_topics = sys.argv[3]
else:
    num_topics = 10

def main():

    # Must use pandas 0.18.1
    # Run these commands from terminal:
    #
    #pip uninstall pandas
    #pip install -I pandas==0.18.1
    #
    # and then:
    print(pd.__version__)

    # Optionally use a tokenizer

    #tokenizer = RegexpTokenizer(r'\w+')

    # Create p_stemmer of class PorterStemmer

    p_stemmer = PorterStemmer()

    # Read data into pandas dataframe

    df = pd.read_csv(input_file)
 
    # Creating a set of Stopwords
    
    stops = set(nltk.corpus.stopwords.words("english"))
    stops.add('yes')

    with open("stopwords/extra_stopwords.txt", 'r') as sw:
        extra_stops = sw.readlines()
        extra_stops = [i.strip('\n') for i in extra_stops]

    stops.update(extra_stops)
    p_stemmer = nltk.stem.porter.PorterStemmer()        # Creating the stemmer model


    # In[ ]:

    df.columns = ['interview','department', 'division','role','intro','use_of_data','other_data']


    # In[ ]:

    def cleaner(row):
        if row:
            '''Function to clean the text data and prep for further analysis'''
            #text = row[col].lower()                        # Converts to lower case
            row = sub("\\(.+?\\)", "", row) #added in from cleaner1 function count_words_in_doc_3
            text = row.lower()                              # Converts to lower case
            text = re.sub("[^a-zA-Z]"," ",text)             # Removes punctuation
            text = text.split()                             # Splits the data into individual words
            text = [w for w in text if not w in stops]      # Removes stopwords
            text = [p_stemmer.stem(i) for i in text]        # Stemming (reducing words to their root)
            return text                                     # Function output
        else:
            return None

    assert cleaner('This is an interview ...(Inaudible)') == ['interview']
    assert cleaner('This is an example of stop word removal, stemming, and cleaning..') == ['exampl', 'word', 'remov', 'stem', 'clean']
    assert cleaner(None) == None


    # In[ ]:

    df.columns
    df['combined'] = df['intro'] + df['use_of_data'] + df['other_data']


    # In[ ]:

    tokens = df['combined'].apply(cleaner)


    # In[ ]:

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(tokens)
    len(dictionary)


    # In[ ]:

    dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=3000)
    len(dictionary)


    # In[ ]:

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(token) for token in tokens]


    # In[ ]:

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word = dictionary,
        passes=10
    )


    # In[ ]:

    print(ldamodel)


    # In[ ]:

    print(ldamodel.print_topics(num_words=20))


    # In[ ]:

    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)        # Visualise LDA Model


    # In[ ]:

    pyLDAvis.save_html(data=vis,fileobj= vis_file_name)     # Save html output
    print(vis_file_name + ' file created!')

if __name__ == '__main__':
    main()
