#Essentials
import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import pickle
# Speech to text and text to speech
from gtts import gTTS
from io import BytesIO
import pygame
import speech_recognition as sr
# Deep Learning
import tflearn
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
#Dashboard and Deployment
import streamlit as st

st.header('Talk With Danny')


with open('intents.json') as file:
    data=json.load(file)
# data['intents']


################# MAIN FUNCTION #############################
#########################################################################

with open("data.pickle","rb") as f:
    words,labels,training,output = pickle.load(f)


############################MODEL LOADING################################
##########################################################################################


model=tf.keras.models.load_model('model')


######################## MAKING PREDICTIONS AND GETTING OUTPUT ##################
#################################################################################

## bag of words function
def bag_of_words (s,words):
    bag=[0 for _ in range(len(words))]

    s_words= nltk.word_tokenize(s)
    stemmer=LancasterStemmer()
    s_words = [stemmer.stem(words.lower()) for words in s_words]

    for se in s_words:
        for i,w in enumerate (words):
            if w == se:
                bag[i]=1


    return np.array([bag])

#### Output as audio function 
def speak(text, language='en'):
    mp3_fo=BytesIO()
    tts =gTTS(text, lang=language)
    tts.write_to_fp(mp3_fo)
    return mp3_fo
pygame.init()
pygame.mixer.init()


def chat():
    st.markdown('Start talking with the bot! (type quit to stop)')
    counter=0

    counter+=1
    inp = st.text_input("You:" , key=f"input")
    


    
    results=model.predict(bag_of_words(inp, words))
    
    results_index= np.argmax(results)
    tag=labels[results_index]
    
    for tg in data['intents']:
        if tg['tag']==tag:
            responses=tg['responses']
    out=random.choice(responses)
    st.write(f'Danny: {out}')
    
    sound=speak(out)
    pygame.mixer.music.load(sound, 'mp3')
    pygame.mixer.music.play()
    st.session_state['text']=''
chat()






























