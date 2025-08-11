# Resturant-Reviews
This project is a Flask-based web application that analyzes restaurant reviews and classifies them as Positive 😊 or Negative 😞 using Natural Language Processing (NLP) and Machine Learning.

Features
Text Preprocessing: Cleans user input by removing non-alphabetic characters, converting to lowercase, removing stopwords, and applying stemming.

Bag of Words Model: Uses CountVectorizer to convert text into numerical features.

Machine Learning Model: Implements a Multinomial Naive Bayes classifier for sentiment prediction.

Interactive Web Interface: Users can enter a review and instantly receive feedback.

Tech Stack
Backend: Flask (Python)

NLP: NLTK for stopword removal and stemming

Machine Learning: Scikit-learn (CountVectorizer, MultinomialNB)

Frontend: HTML (via Flask templates)

Dataset: Restaurant_Reviews.tsv (Tab-separated restaurant reviews labeled as positive or negative)

How It Works
The dataset is preprocessed and transformed into numerical vectors using a Bag of Words model.

A Naive Bayes classifier is trained on the transformed dataset.

User input is processed in the same way and fed into the trained model.

The application displays whether the review is Positive or Negative.
