
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from itertools import islice
import string
import warnings
warnings.filterwarnings("ignore")


# Function to check differences and assign flag numbers
def check_diff(group):
    # Sort by id to ensure the order
    group = group.sort_values(by='id')
    
    # Calculate the difference between consecutive values
    diffs = group['id'].diff()
    
    # Initialize a flag counter
    flag_counter = 0
    flags = []

    # Iterate through differences and increment flag counter when needed
    for diff in diffs:
        if pd.isna(diff) or diff <= 1:
            flags.append(flag_counter)
        else:
            flag_counter += 1
            flags.append(flag_counter)

    group['Flag'] = flags
    return group


## uptake

stop_words = set(stopwords.words('english'))
additional_stopwords = {'teacher:', 'student:'}
stop_words.update(additional_stopwords)

def remove_punc(word_string):
    word_string = word_string.lower()
    word_list = word_string.split()
    filtered_words = [word for word in word_list if word not in additional_stopwords]
    translator = str.maketrans(' ', ' ', string.punctuation)
    word_string = ' '.join(filtered_words).translate(translator)
    word_string = re.sub("  ", " ", word_string)
    return word_string.strip()


def remove_stop_words(word_string):
    word_string = word_string.lower()

    word_list = word_string.split()
    filtered_words = [word for word in word_list if word not in stop_words]

    # Create a translation table that maps each punctuation character to a space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    word_string = ' '.join(filtered_words).translate(translator)
    word_string = re.sub(" +", " ", word_string)  # Modified to handle multiple consecutive spaces
    return word_string.split()

def is_sublist(source, target):
    slen = len(source)
    return any(all(item1 == item2 for (item1, item2) in zip(source, islice(target, i, i+slen))) for i in range(len(target) - slen + 1))

def long_substr_by_word(data):
    subseq = []
    data_seqs = [s.split(' ') for s in data]
    if len(data_seqs) > 1 and len(data_seqs[0]) > 0:
        for i in range(len(data_seqs[0])):
            for j in range(len(data_seqs[0])-i+1):
                if j > len(subseq) and all(is_sublist(data_seqs[0][i:i+j], x) for x in data_seqs):
                    subseq = data_seqs[0][i:i+j]
    return subseq

def common_words(dial1, dial2):
    dial1 = set(remove_stop_words(dial1))
    dial2 = set(remove_stop_words(dial2))
    return len(dial1.intersection(dial2))

def percent_dial1(dial1, dial2):
    dial1 = set(remove_stop_words(dial1))
    dial2 = set(remove_stop_words(dial2))
    if not dial1:
        return 0
    return len(dial1.intersection(dial2)) / len(dial1)
    


def percent_dial2(dial1, dial2):
    dial1 = set(remove_stop_words(dial1))
    dial2 = set(remove_stop_words(dial2))
    if not dial2:
        return 0
    return len(dial1.intersection(dial2)) / len(dial2)
    

