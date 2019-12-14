import numpy as np
from flask import Flask, request, jsonify, render_template, redirect,url_for
import pickle
from sklearn.externals import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')
nltk.download("stopwords")
nltk.download('wordnet')
punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

app = Flask(__name__)
#
svc_model = pickle.load(open('best_svc.pickle', 'rb'))
tfidf = pickle.load(open('tfidf.pickle', 'rb'))
#model = joblib.load('model.pkl') 
output = ""
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]

    global output
    output = get_news(int_features)
    output = output.upper()
    #return render_template('index.html', prediction_text='The House Price for the selected options would be : Rs. {}'.format(output))
    return redirect(url_for('showresult'))

@app.route('/result')
def showresult():
    return render_template('showresult.html', prediction_text='Given Article is related to  {}'.format(output))

def get_news(int_features):
    #user_input = request.form.value()
    #
    #news_contents = "ime Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business"
    news_contents = str(int_features)
    df_features = pd.DataFrame(
         {'Content': [news_contents] 
        })
    features = create_features_from_df(df_features)
    prediction_text = predict_from_features(features)

    return " ".join(prediction_text)

def create_features_from_df(df):
    
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
        
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    
    wordnet_lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row]['Content_Parsed_4']
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    
    df['Content_Parsed_5'] = lemmatized_text_list
    
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
        
    df = df['Content_Parsed_6']
    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features

def predict_from_features(features):
        
    # Obtain the highest probability of the predictions for each article
    predictions_proba = svc_model.predict_proba(features).max(axis=1)    
    
    # Predict using the input model
    predictions_pre = svc_model.predict(features)

    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob, cat in zip(predictions_proba, predictions_pre):
        if prob > .65:
            predictions.append(cat)
        else:
            predictions.append(5)

    # Return result
    categories = [get_category_name(x) for x in predictions]
    
    return categories
category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4,
    'other':5
}

def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category

if __name__ == "__main__":
    app.run(debug=True)