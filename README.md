# SENTIMENT-ANALYSIS

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 2. Download NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 3. Load Dataset
# Ensure you have downloaded the IMDB Kaggle dataset from:
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Save as "IMDB Dataset.csv" in your working directory

df = pd.read_csv("IMDB _Dataset.csv")  # Column names: ['review', 'sentiment']

# Convert sentiment to binary values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 4. Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing (may take a few minutes)
df['cleaned_review'] = df['review'].apply(preprocess_text)

# 5. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['sentiment'].values

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Prediction & Evaluation
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.show()

# 9. Insights
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.show()

# 10. Test on New Input
new_review = "The movie was absolutely wonderful and inspiring."
cleaned = preprocess_text(new_review)
vector = tfidf.transform([cleaned]).toarray()
pred = model.predict(vector)
sentiment = "Positive" if pred[0] == 1 else "Negative"
print(f"\nSentiment for the review: {sentiment}")

## OUTPUT

[nltk_data] Downloading package punkt to

[nltk_data]     C:\Users\chait\AppData\Roaming\nltk_data...

[nltk_data]   Package punkt is already up-to-date!

[nltk_data] Downloading package stopwords to

[nltk_data]     C:\Users\chait\AppData\Roaming\nltk_data...

[nltk_data]   Package stopwords is already up-to-date!

[nltk_data] Downloading package wordnet to

[nltk_data]     C:\Users\chait\AppData\Roaming\nltk_data...

[nltk_data]   Package wordnet is already up-to-date!

Accuracy: 88.62%


Classification Report:

              precision    recall  f1-score   support

           0       0.90      0.87      0.88      4961
           1       0.88      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

<Figure size 600x400 with 2 Axes>
