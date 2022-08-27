from flask import Flask, render_template, request, flash
import pickle
import tensorflow
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import re

# Load the tensorflow model
model = tensorflow.keras.models.load_model('final_model.h5')
tokenizer = pickle.load(open('final_tokenizer.pickle', 'rb'))

# Create Clean Text Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[a-zA-Z0-9_]*', '', text)
    text = re.sub(r'https?://[a-zA-Z0-9./]*', '', text)
    text = re.sub('[^a-zA-z0-9\s]','',text)
    print(text)
    return text


def predict_sentiment(text):
    twt = [clean_text(text)]
    #vectorizing the tweet by the pre-fitted tokenizer instance
    twt = tokenizer.texts_to_sequences(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=40, dtype='int32', value=0)
    print(twt)
    sentiment = model.predict(twt,batch_size=1)[0]
    if(np.argmax(sentiment) == 0):
        return("Négatif")
    elif (np.argmax(sentiment) == 1):
        return("Positif")

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'some_secret'
@app.route('/predict')

def index():
    flash("Inconnu", 'prediction')
    flash("", 'tweet')
    return render_template('index.html')

@app.route('/predicted', methods=['POST', 'GET'])
def predicted():
    if request.method == 'POST':
        twt = request.form['tweet']
        flash(request.form['tweet'], 'tweet')
        flash(predict_sentiment(twt), 'prediction')
        return render_template('index.html')
