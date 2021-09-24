#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:14:48 2021

@author: astafyevayana
"""

import numpy as np 
import os
import pickle
import json
import nltk
import random
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from keras import callbacks 
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
lemmatizer = WordNetLemmatizer()

# Upload API
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def upload_file():
    
    # downloading the costumer's file
       
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save( filename)
            print("saved file successfully")
            
            #return redirect(filename)            
            file_path =  filename
            # format of file
            select = request.form.get('comp_select')
            # costemer's initialization of chatbot
            select1 = request.form.get('comp_select1')
           
################## Chatbot ####################################################
            # for now code run just for .json files
            if select=='.json':                
                if select1=='Dialogue':
                    # save initialization of chatbot for next html window
                    number = 1
                    
                    with open('ID_bot.txt', 'w') as f:
                        f.write('%d' % number)
                    # save path of downloaded file    
                    with open('ID_bot_path1.txt', 'w') as f:
                        f.write(file_path)
                
                    data1= open(file_path).read()
                    data = json.loads(data1)
                            
                    training_sentences = []
                    training_labels = []
                    labels = []
                    responses = []
                                        
                    for intent in data['intents']:
                        for pattern in intent['patterns']:
                            training_sentences.append(pattern)
                            training_labels.append(intent['tag'])
                        responses.append(intent['responses'])
                        
                        if intent['tag'] not in labels:
                            labels.append(intent['tag'])
                            
                    num_classes = len(labels)
                    labels1 = sorted(list(set(labels)))
                    
                    import pickle
                    # save classes for speaking with chatbot              
                    pickle.dump(labels1, open("classes.pkl", "wb"))
                    
                    lbl_encoder = LabelEncoder()
                    lbl_encoder.fit(training_labels)
                    training_labels = lbl_encoder.transform(training_labels)
                    
                    vocab_size = 1000
                    embedding_dim = 16
                    max_len = 200
                    oov_token = "<OOV>"
                    
                    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
                    tokenizer.fit_on_texts(training_sentences)
                    word_index = tokenizer.word_index
                    
                    sequences = tokenizer.texts_to_sequences(training_sentences)
                    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
                   
                    # save words for speaking with chatbot     
                    wwords = sorted(list(set(word_index.keys() )))
                    pickle.dump(wwords[1:], open("words.pkl", "wb"))
                    
                    # we can build model inside this code by:
                    '''
                    model = Sequential()
                    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
                    model.add(GlobalAveragePooling1D())
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(num_classes, activation='softmax'))
                    '''
                    # load model
                    model = load_model("chat_model_dialogue")
                    # important to compile every time the model
                    model.compile(loss='sparse_categorical_crossentropy', 
                                  optimizer='adam', metrics=['accuracy'])
                    
                    #model.summary()                    
                    epochs = 200
                    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
                    
                    # to save the trained model
                    #model.save("chat_model_dialogue")
                    
                    import pickle
                    
                    # to save the fitted tokenizer
                    with open('tokenizer.pickle', 'wb') as handle:
                        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    # to save the fitted label encoder
                    with open('label_encoder.pickle', 'wb') as ecn_file:
                        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
                
                if select1=='Voyage':
                    # save initialization of chatbot for next html window
                    number = 2
                    with open('ID_bot.txt', 'w') as f:
                        f.write('%d' % number)
                    # save path of downloaded file  
                    with open('ID_bot_path2.txt', 'w') as f:
                        f.write(file_path)    
                    # open downloaded data              
                    data1= open(file_path).read()                                                       
                    data = json.loads(data1)
                           
                    training_sentences = []
                    training_labels = []
                    labels = []
                    responses = []                    
                    
                    for intent in data['intents']:
                        for pattern in intent['patterns']:
                            training_sentences.append(pattern)
                            training_labels.append(intent['tag'])
                        responses.append(intent['responses'])
                        
                        if intent['tag'] not in labels:
                            labels.append(intent['tag'])
                            
                    num_classes = len(labels)
                    labels1 = sorted(list(set(labels)))
                    
                    import pickle
                    # save classes for speaking with chatbot               
                    pickle.dump(labels1, open("classes.pkl", "wb"))
                    
                    lbl_encoder = LabelEncoder()
                    lbl_encoder.fit(training_labels)
                    training_labels = lbl_encoder.transform(training_labels)
                    
                    vocab_size = 1000
                    embedding_dim = 16
                    max_len = 200
                    oov_token = "<OOV>"
                    
                    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
                    tokenizer.fit_on_texts(training_sentences)
                    word_index = tokenizer.word_index
                    
                    sequences = tokenizer.texts_to_sequences(training_sentences)
                    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
                   
                    # save words for speaking with chatbot     
                    wwords = sorted(list(set(word_index.keys() )))
                    pickle.dump(wwords[1:], open("words.pkl", "wb"))
                    
                    # we can build model inside this code by:
                    '''
                    model = Sequential()
                    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
                    model.add(GlobalAveragePooling1D())
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(num_classes, activation='softmax'))
                    model.compile(loss='sparse_categorical_crossentropy', 
                                  optimizer='adam', metrics=['accuracy'])
                    model.summary()
                    '''
                    model = load_model("chat_model_voyage")
                    model.compile(loss='sparse_categorical_crossentropy', 
                                  optimizer='adam', metrics=['accuracy'])
                    
                    
                    
                    
                    
                    epochs = 200
                    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
                    # to save the trained model
                    #model.save("chat_model_voyage")
                    
                    import pickle
                    
                    # to save the fitted tokenizer
                    with open('tokenizer.pickle', 'wb') as handle:
                        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    # to save the fitted label encoder
                    with open('label_encoder.pickle', 'wb') as ecn_file:
                        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
                
                if select1=='Anecdote':
                    # save initialization of chatbot for next html window
                    number = 3
                    with open('ID_bot.txt', 'w') as f:
                        f.write('%d' % number)
                    # save path of downloaded file  
                    with open('ID_bot_path3.txt', 'w') as f:
                        f.write(file_path)    
                    # open downloaded data
                    data1= open(file_path).read()
                    data = json.loads(data1)
                    
        
                    training_sentences = []
                    training_labels = []
                    labels = []
                    responses = []
                                  
                    for intent in data['intents']:
                        for pattern in intent['patterns']:
                            training_sentences.append(pattern)
                            training_labels.append(intent['tag'])
                        responses.append(intent['responses'])
                        
                        if intent['tag'] not in labels:
                            labels.append(intent['tag'])
                            
                    num_classes = len(labels)
                    labels1 = sorted(list(set(labels)))
                    
                    import pickle
                    # save classes for speaking with chatbot               
                    pickle.dump(labels1, open("classes.pkl", "wb"))
             
                    lbl_encoder = LabelEncoder()
                    lbl_encoder.fit(training_labels)
                    training_labels = lbl_encoder.transform(training_labels)
                    
                    vocab_size = 1000
                    embedding_dim = 16
                    max_len = 200
                    oov_token = "<OOV>"
                    
                    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
                    tokenizer.fit_on_texts(training_sentences)
                    word_index = tokenizer.word_index
                    
                    sequences = tokenizer.texts_to_sequences(training_sentences)
                    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
                   
                    # save words for speaking with chatbot     
                    wwords = sorted(list(set(word_index.keys() )))
                    pickle.dump(wwords[1:], open("words.pkl", "wb"))
                    
                    # we can build model inside this code by:
                    '''
                    model = Sequential()
                    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
                    model.add(GlobalAveragePooling1D())
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(num_classes, activation='softmax'))
                    model.compile(loss='sparse_categorical_crossentropy', 
                                  optimizer='adam', metrics=['accuracy'])
                    model.summary()
                    '''
                    model = load_model("chat_model_anecdote")
                    model.compile(loss='sparse_categorical_crossentropy', 
                                  optimizer='adam', metrics=['accuracy'])
               
                    epochs = 200
                    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
                    # to save the trained model
                    #model.save("chat_model_anecdote")
                    
                    import pickle                   
                    # to save the fitted tokenizer
                    with open('tokenizer.pickle', 'wb') as handle:
                        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    # to save the fitted label encoder
                    with open('label_encoder.pickle', 'wb') as ecn_file:
                        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
                
               
    return render_template('upload_file3.html', data=[{'name':'.txt'}, {'name':'.json'}], data3=[{'name':'t'}, {'name':'Dialogue'},{'name':'Voyage'},{'name':'Anecdote'}])

#chatbot html windows
@app.route("/test", methods = ['GET', 'POST'])
def test():        
    return render_template("index_2.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    
    with open('ID_bot.txt') as f:
        id_bot = f.readlines() 
    
    print('id_bot',id_bot)
  
    if id_bot==['1']:
    
        model = load_model("chat_model_dialogue")
   
        with open('ID_bot_path1.txt') as f:
            id_bot_path1 = f.readlines()[0]
  
        data1= open(id_bot_path1).read()                    
        intents = json.loads(data1)
            
    if id_bot==['2']:
    
        model = load_model("chat_model_voyage")
        with open('ID_bot_path2.txt') as f:
            id_bot_path2 = f.readlines()[0]
  
    
        data1= open(id_bot_path2).read()                    
        intents = json.loads(data1)
        
    if id_bot==['3']:
    
        model = load_model("chat_model_anecdote")
        with open('ID_bot_path3.txt') as f:
            id_bot_path3 = f.readlines()[0]
  
    
        data1= open(id_bot_path3).read()                    
        intents = json.loads(data1)
   
    msg = request.form["msg"]
    #checks is a user has given a name, in order to give a personalized feedback
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    #if no name is passed execute normally
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res


# chat functionalities
def clean_up_sentence(sentence):
   
    
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern      
    words = pickle.load(open("words.pkl", "rb"))
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    
    with open('ID_bot.txt') as f:
        id_bot = f.readlines() 
    
    print('id_bot',id_bot)
    # for different task-different models
    if id_bot==['1']:
    
        model = load_model("chat_model_dialogue")
   
    if id_bot==['2']:
    
        model = load_model("chat_model_voyage")
    
    
    if id_bot==['3']:
    
        model = load_model("chat_model_anecdote")
       
    model = load_model("chat_model_anecdote")
       
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.12
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]    
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result 

# make API work
if __name__ == "__main__":
    app.run(host='localhost')    