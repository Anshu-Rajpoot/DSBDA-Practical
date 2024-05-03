import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample document
sample_document = """
Text analytics involves processing and analyzing large amounts of unstructured text data to derive meaningful insights. 
It includes various techniques such as tokenization, part-of-speech tagging, stop words removal, stemming, and lemmatization.
TF-IDF (Term Frequency-Inverse Document Frequency) is a common method used to represent text data in machine learning.
"""

# Tokenization
tokens = word_tokenize(sample_document)

# Stop words removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Stemming
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

# Lemmatization
wnl = WordNetLemmatizer()
lemmatized_tokens = [wnl.lemmatize(word) for word in filtered_tokens]

# Calculate Term Frequency (TF)
tf_vectorizer = TfidfVectorizer(use_idf=False)
tf_matrix = tf_vectorizer.fit_transform([sample_document])
tf_terms = tf_vectorizer.get_feature_names_out()

# Calculate Inverse Document Frequency (IDF)
idf_vectorizer = TfidfVectorizer(use_idf=True)
idf_matrix = idf_vectorizer.fit_transform([sample_document])
idf_terms = idf_vectorizer.get_feature_names_out()
idf_values = idf_matrix.data

# Print results
print("Original Text:", sample_document)
print("\nTokenization:", tokens)
print("\nStop Words Removal:", filtered_tokens)
print("\nStemming:", stemmed_tokens)
print("\nLemmatization:", lemmatized_tokens)
print("\nTerm Frequency (TF):")
for term, tf_value in zip(tf_terms, tf_matrix.toarray()[0]):
    print(f"{term}: {tf_value}")
print("\nInverse Document Frequency (IDF):")
for term, idf_value in zip(idf_terms, idf_values):
    print(f"{term}: {idf_value}")
