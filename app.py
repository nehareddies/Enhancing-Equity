from flask import Flask, render_template, request
import pickle
from model import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
model = pickle.load(open('discrimination.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    discrimination = str(request.form['text'])
    discrimination = discrimination.lower()
    discrimination = re.sub(r'[^a-zA-Z\s]', '', discrimination)
    discrimination = re.sub(r'\d+', '', discrimination)
    tokens = word_tokenize(discrimination)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)

    data = [{'new_text': preprocessed_text}]
    updated = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['new_text'])
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    texts = vectorizer.transform(updated['new_text'])
    result = model.predict(texts)

    if result == "Gender Discrimination":
        result = "The Text Shows Gender Discrimination"
    elif result == "Appearance Discrimination":
        result = "The Text Shows Appearance Discrimination"
    elif result == "Social Class Discrimination":
        result = "The Text Shows Social Class Discrimination"
    else:
        result = "The Text Does Not Show Any Discrimination!"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
