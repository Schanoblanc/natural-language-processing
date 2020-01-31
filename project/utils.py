import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype
    embeddings = {}
    print('debug hit')
    for line in open(embeddings_path,'r',encoding="utf-8"):
        line = line.replace("\n","") # remove the last \n in the end of each line
        terms = line.split("\t") # tsv is tab-separated-value so split by tab
        word_key = terms[0] # the first term is the word
        string_vector = terms[1:]
        vector = np.array(string_vector,dtype = np.float32)
        embeddings[word_key] = vector
    
    first_key = next(iter(embeddings.keys()))
    first_vector = embeddings[first_key]
    embeddings_dim = len(first_vector)
    
    return embeddings,embeddings_dim
    ########################
    #### YOUR CODE HERE ####
    ########################

    # remove this when you're done
    #raise NotImplementedError(
    #    "Open utils.py and fill with your code. In case of Google Colab, download"
    #    "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
    #    "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.
    ## copy from week 3
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    words = question.split(' ')
    count = 0
    result = np.zeros([1,dim])
    for word in words:
        if word in embeddings:
            count += 1
            result = result + embeddings[word]
    
    if count > 0:
        result = result / count
    return result
    ########################
    #### YOUR CODE HERE ####
    ########################

    # # remove this when you're done
    # raise NotImplementedError(
    #     "Open utils.py and fill with your code. In case of Google Colab, download"
    #     "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
    #     "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
