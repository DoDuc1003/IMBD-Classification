import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def padding(text, max_len):
    words = text.split(' ')
    if len(words) < max_len:
        words = words + ["<pad>" for _ in range(max_len - len(words))]
    else:
        words = words[0: max_len]
    
    result = " ".join(words)
    return result

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

