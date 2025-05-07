# SENTIMENT-ANALYSIS

COMPANY: CODTECH IT SOLUTIONS

NAME:KHUSHI SHAH

INTERN ID:CT04DL131

DOMAIN:DATA ANALYTICS

DURATION:4 WEEKS

MENTOR:NEELA SANTHOSH

## DESCRIPTION

This Python script implements a complete Natural Language Processing (NLP) pipeline for **sentiment analysis** on movie reviews using the **IMDB dataset**. The process includes text preprocessing, feature extraction using TF-IDF, model training with logistic regression, evaluation, and even prediction on new custom input. It serves as a comprehensive example of applying machine learning to text data.

#### **1. Library Imports and NLTK Resource Downloads**

The script begins by importing standard data science libraries including `pandas`, `numpy`, `matplotlib`, and `seaborn` for data handling and visualization. NLP-specific tools are loaded from the `nltk` library, including tokenizers, stopword lists, and lemmatizers. Scikit-learn modules are used for machine learning model training and evaluation.

The script downloads essential NLTK resources like `punkt`, `stopwords`, and `wordnet`, which are required for tokenization and lemmatization during preprocessing.

#### **2. Loading and Preparing the Dataset**

The IMDB dataset is read from a CSV file (`IMDB_Dataset.csv`) and consists of 50,000 movie reviews labeled as either “positive” or “negative”. These sentiment labels are converted into binary format where “positive” is mapped to 1 and “negative” to 0, enabling binary classification.

#### **3. Text Preprocessing**

A `preprocess_text()` function is defined to clean and normalize the text. Each review is:

* Converted to lowercase.
* Tokenized using `word_tokenize()` from NLTK.
* Filtered to retain only alphabetic words and exclude stopwords.
* Lemmatized using `WordNetLemmatizer` to reduce words to their base form.

This step significantly reduces noise in the data and standardizes input text, which helps the model learn more effectively. The function is applied to all reviews and the results are stored in a new column called `cleaned_review`.

#### **4. TF-IDF Vectorization**

After cleaning, the text is transformed into numerical features using `TfidfVectorizer`, which captures the importance of each word relative to all documents. The vectorizer is limited to the top 5,000 features to reduce dimensionality and improve model efficiency. The result is a matrix `X` containing TF-IDF values, while the target labels are stored in `y`.

#### **5. Train-Test Split and Model Training**

The dataset is split into training and testing sets using an 80-20 ratio. A `LogisticRegression` model is then trained on the training data. Logistic regression is a simple yet effective algorithm for binary classification tasks and performs well on high-dimensional text data.

#### **6. Evaluation**

After training, predictions are made on the test set, and performance is evaluated using:

* **Accuracy Score**: Overall percentage of correct predictions.
* **Classification Report**: Includes precision, recall, and F1-score for both classes.
* **Confusion Matrix**: Visualized using a heatmap to show true/false positives and negatives.

Additionally, a bar plot of sentiment distribution is created to visualize class balance.

#### **7. Custom Input Prediction**

The final part of the script tests the model on a new review. The custom sentence is cleaned using the same preprocessing function, vectorized using the TF-IDF model, and classified by the trained logistic regression model. The predicted sentiment is printed to the console.

## OUTPUT

Accuracy: 88.62%


Classification Report:

              precision    recall  f1-score   support

           0       0.90      0.87      0.88      4961
           1       0.88      0.90      0.89      5039

    accuracy                           0.89     10000

    
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000

![image](https://github.com/user-attachments/assets/2d7f7a7a-b3ac-4586-af71-b57ba319cf92)

![image](https://github.com/user-attachments/assets/bebcf1c8-8a9e-4ab3-9fcf-75bd32434052)


