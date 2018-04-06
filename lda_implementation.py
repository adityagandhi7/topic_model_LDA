## Implementing LDA in Python using the "A Million Headlines" dataset provided by the ABC (Australia)

# importing the relevant libraries

import pandas as pd
import numpy as np
from IPython.display import display
#from tqdm import tqdm
# Collections provides specialized datatypes
from collections import Counter
# ast stands for Abstract Syntax Trees, a way to better parse code into an AST that represents the code
import ast
import matplotlib.pyplot as plt
# For compatibility with MATLAB commands
import matplotlib.mlab as mlab
import seaborn as sb

# Reading in the data, and then also reading the dates in the correct format
raw_data= pd.read_csv('abcnews-date-text.csv', parse_dates=[0], infer_datetime_format=True)
# Indexing the data to make the date the index
reindexed_data = raw_data['headline_text']
reindexed_data.index = raw_data['publish_date']


# Define helper function get_top_n_words
if __name__ == '__main__':
    def get_top_n_words(n_top_words, count_vectorizer, text_data):
        '''returns a tuple of the top n words in a sample and their accompanying counts, given a CountVectorizer object and text sample'''
        vectorized_headlines = count_vectorizer.fit_transform(text_data.as_matrix())

        vectorized_total = np.sum(vectorized_headlines, axis=0)
        word_indices = np.fliplr(np.argsort(vectorized_total)[0, :])
        word_values = np.fliplr(np.sort(vectorized_total)[0, :])

        word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
        for i in range(n_top_words):
            word_vectors[i, word_indices[0, i]] = 1

        words = [word[0].encode('ascii').decode('utf-8') for word in count_vectorizer.inverse_transform(word_vectors)]

        return (words, word_values[0, :n_top_words].tolist()[0])

# We develop the list of top words used across all one million headlines. Stop words are omitted

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
words, word_values = get_top_n_words(n_top_words=20, count_vectorizer=count_vectorizer, text_data=reindexed_data)

# Now we will implement POS (part-of-speech) tagging on the raw data. And we will also obtain the total number of words and the mean number of words per headline
# Note, we cannot run this right now because of the import toolbox issues. textblob is unable to be installed

from textblob import TextBlob

while True:
    try:
        tagged_headlines = pd.read_csv('abcnews-pos-tagged.csv', index_col=0)
        word_counts = []
        pos_counts = {}

        for headline in tagged_headlines[u'tags']:
            headline = ast.literal_eval(headline)
            word_counts.append(len(headline))
            for tag in headline:
                if tag[1] in pos_counts:
                    pos_counts[tag[1]] += 1
                else:
                    pos_counts[tag[1]] = 1

    except IOError:
        tagged_headlines = [TextBlob(reindexed_data[i]).pos_tags for i in range(reindexed_data.shape[0])]

        tagged_headlines = pd.DataFrame({'tags':tagged_headlines})
        tagged_headlines.to_csv('abcnews-pos-tagged.csv')
        continue
    break

# Preprocessing before building the LDA model
# We will use a sample of the dataset and then vectorize the text headlines, i.e., convert to an n x K document-term matrix
# K is the number of distinct words across n headlines in the sample. The sample has fewer stop words and a limit on max_fatures.

small_count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
small_text_sample = reindexed_data.sample(n=10000, random_state=0).as_matrix()
print('Headline before vectorization: ', small_text_sample[123])

# We get a very high-rank and sparse training data, and can now implement a clustering algorithm
small_document_term_matrix = small_count_vectorizer.fit_transform(small_text_sample)
print('Headline after vectorization: \n', small_document_term_matrix[123])

# Setting the number of initial topics to 8
n_topics= 8

# Creating the LDA Model

from sklearn.decomposition import LatentDirichletAllocation

# Note if the data size is large, we use learning_method='online'

lda_model = LatentDirichletAllocation(n_topics=n_topics, learning_method='online', random_state=0, verbose=0)
lda_topic_matrix = lda_model.fit_transform(small_document_term_matrix)

# Now that we have the topic matrix. We need to get the arg mx (or the topics with the max probabilites).
# We also need to get the most ferequent words in this topic
# For this we define two helper functions, get_keys and keys_to_counts

# Define helper functions
def get_keys(topic_matrix):
    '''returns an integer list of predicted topic categories for a given topic matrix'''
    keys = []
    for i in range(topic_matrix.shape[0]):
        keys.append(topic_matrix[i].argmax())
    return keys

def keys_to_counts(keys):
    '''returns a tuple of topic categories and their accompanying magnitudes for a given list of keys'''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

lda_keys = get_keys(lda_topic_matrix)
lda_categories, lda_counts = keys_to_counts(lda_keys)

# Once we have the LDA Counts and categories, we again use the get_top_n_words function to get the most frequent words in each topic

# Define helper functions - we have redefined this function
def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
    '''returns a list of n_topic strings, where each string contains the n most common
        words in a predicted category, in order'''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flipud(np.argsort(temp_vector_sum)[0][-n:])
        top_word_indices.append(top_n_word_indices)
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words

# This gives the 8 topics that are associated with the words
top_n_words_lda = get_top_n_words(10, lda_keys, small_document_term_matrix, small_count_vectorizer)
for i in range(len(top_n_words_lda)):
    print("Topic {}:".format(i), top_n_words_lda[i])

# Looking at the top 3 words instead of the top 10 words

top_3_words = get_top_n_words(3, lda_keys, small_document_term_matrix, small_count_vectorizer)
for i in range(len(top_3_words)):
    print("Topic {}:".format(i), top_3_words[i])

# Now that we have obtained a list of topics for a small dataset, we will consider a larger set with 100000 records

big_sample_size = 100000
big_count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
big_text_sample = reindexed_data.sample(n=big_sample_size, random_state=0).as_matrix()
big_document_term_matrix = big_count_vectorizer.fit_transform(big_text_sample)

# Now we run a larger LDA model for the document_term_matrix
big_lda_model = LatentDirichletAllocation(n_topics=n_topics, learning_method='online', verbose=1)
big_lda_model.fit(big_document_term_matrix)


# Now since we have a big topic model, we will need to pass the entire dataset of one million in this model. And we will sort by years

# We create a data structure. i.e. list, that has data grouped by year in each element of the list
yearly_data = []
for i in range(2003,2017+1):
    yearly_data.append(reindexed_data['{}'.format(i)].as_matrix())

# We create a topic matrix for each year based on the big_lda_model fit
yearly_topic_matrices = []
for year in yearly_data:
    document_term_matrix = big_count_vectorizer.transform(year)
    topic_matrix = big_lda_model.transform(document_term_matrix)
    yearly_topic_matrices.append(topic_matrix)

# Now that we have the probabilities for each year, we plan to get the keys using the argmax in the get_keys function
yearly_keys = []
for topic_matrix in yearly_topic_matrices:
    yearly_keys.append(get_keys(topic_matrix))

# Similarly we also get the counts for these keys
yearly_counts = []
for keys in yearly_keys:
    categories, counts = keys_to_counts(keys)
    yearly_counts.append(counts)

# Now we plot the count per year for each topic
yearly_topic_counts = pd.DataFrame(np.array(yearly_counts), index=range(2003,2017+1))
yearly_topic_counts.columns = ['Topic {}'.format(i) for i in range(n_topics)]
print(yearly_topic_counts)

# After the count, we get the most frequent works for each topic

complete_keys = [key for year in yearly_keys for key in year]
complete_topic_matrix = np.vstack(yearly_topic_matrices)
complete_document_term_matrix = big_count_vectorizer.transform(reindexed_data.as_matrix())

top_n_words = get_top_n_words(10, complete_keys, complete_document_term_matrix, big_count_vectorizer)

for i in range(len(top_n_words)):
    print('Topic {}: '.format(i), top_n_words[i])


# Now visualizing this in graph format
labels = ['Topic {}: \n '.format(i) + ' '.join([topic.split() for topic in top_n_words][i][:3]) for i in range(n_topics)]

fig, ax = plt.subplots(figsize=(14,10))
sb.heatmap(yearly_topic_counts, xticklabels=labels, cmap="YlGnBu", ax=ax)













