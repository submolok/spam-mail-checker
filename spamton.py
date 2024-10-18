import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import string

# Load the dataset (change path as needed)
df = pd.read_csv('email_text.csv')

# Download necessary NLTK corpora (if you haven't already)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer and stop words
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess the text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # Remove stop words and stem
    return ' '.join(tokens)

# Fill missing values with an empty string
df['text'] = df['text'].fillna('')

# Now apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Train the model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
