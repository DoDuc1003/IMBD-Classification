import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess




def corpus_length_statistic(corpus):
    lengths = [len(doc) for (doc, _) in corpus]
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    total_documents = len(corpus)
    
    print("Minimum length:", min_length)
    print("Maximum length:", max_length)
    print("Average length:", avg_length)
    print("Total documents:", total_documents)
    
    return min_length, max_length, avg_length, total_documents

