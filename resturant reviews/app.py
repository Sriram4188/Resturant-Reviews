from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download once if not already
nltk.download('stopwords')

# Load and preprocess the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

corpus = []
ps = PorterStemmer()

for review in dataset['Review']:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Feature extraction
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked']

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        input_review = request.form['review']
        cleaned = re.sub('[^a-zA-Z]', ' ', input_review)
        cleaned = cleaned.lower()
        cleaned = cleaned.split()
        cleaned = [ps.stem(word) for word in cleaned if word not in set(stopwords.words('english'))]
        cleaned = ' '.join(cleaned)
        vector = cv.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]
        result = 'Positive 😊' if prediction == 1 else 'Negative 😞'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
