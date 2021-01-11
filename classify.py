# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 00:29:55 2021

@author: Ria Sharma

"""

from flask import Flask, render_template, request

Flask_App = Flask(__name__) 

@Flask_App.route('/', methods=['GET'])
def index():

    return render_template('classify.html')

@Flask_App.route('/Sentiment/', methods=['POST'])
def sentiment_result():
    """Route where we send calculator form input"""
    import keras
    model = keras.models.load_model("model.h5")
    model.summary()
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    SENTIMENT_THRESHOLDS = (0.4, 0.7)
    def decode_sentiment(score, include_neutral=True):
       if include_neutral:        
           label = NEUTRAL
           if score <= SENTIMENT_THRESHOLDS[0]:
               label = NEGATIVE
           elif score >= SENTIMENT_THRESHOLDS[1]:
               label = POSITIVE
    
           return label
       else:
           return NEGATIVE if score < 0.5 else POSITIVE
    import time
    import pickle
    with open("tokenizer.pkl", 'rb') as pickle_file:
       tokenizer = pickle.load(pickle_file)
    from keras.preprocessing.sequence import pad_sequences
    SEQUENCE_LENGTH = 300
    def predict(text, include_neutral=True):
       start_at = time.time()
       # Tokenize text
       x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
       # Predict
       score = model.predict([x_test])[0]
       # Decode sentiment
       label = decode_sentiment(score, include_neutral=include_neutral)
    
       return {"label": label, "score": float(score),
          "elapsed_time": time.time()-start_at}

    text_input = request.form['Input1']
    try:
        input1 = str(text_input)

        return render_template(
            'classify.html',
            input1=input1,
            result=predict(input1)["score"],
            calculation_success=True
        )  
    except ValueError:
        return render_template(
            'classify.html',
            input1=text_input,
            result="Bad Input",
            calculation_success=False,
            error="Cannot able to predict the sentiment!"
        )

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run()
