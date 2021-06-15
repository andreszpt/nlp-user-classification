import numpy as np
import pandas as pd
from os import path
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras import Sequential
from keras.layers import Embedding
from keras.layers import Embedding, Dense, Flatten, Reshape, Dropout, Conv1D, GlobalMaxPooling1D
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


dirname = path.dirname(__file__)
DATA_DIRECTORY = path.join(dirname, '..','data')

def read_hate_file(DATA_DIRECTORY, file):
    csv = pd.read_csv(
        filepath_or_buffer=path.join(DATA_DIRECTORY, file),
        sep='\t',
        header=None)[0]
    return csv


def read_sentiment_file(DATA_DIRECTORY, file):
    csv = pd.read_csv(
        path.join(DATA_DIRECTORY, file),
        names=['polarity', 'id', 'date', 'query', 'user', 'text'],
        encoding='latin-1')
    return csv


def convert_string(data):
    for i in range(len(data)):
        data[i] = str(data[i])
    return data


def clean_data(data):
    data = data.str.lower()
    data = data.apply(lambda x: re.sub(r'http\S+', '', x))
    data = data.apply(lambda x: re.sub(r'@\S+', '', x))
    data = data.apply(lambda x: re.sub(r'&\S+', '', x))
    data = data.apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    data = data.apply(lambda x: " ".join(re.split(r"\s+", x)))
    return data



def tokenize(X_train, X_val, X_test):
    tk = Tokenizer()
    # Update on the vocabulary based on the training set
    tk.fit_on_texts(X_train)
    # word_index is a dict that contains words (key) and index (value), sorted by frequency
    word_index = tk.word_index
    vocab_size = len(word_index) + 1
    return (tk, word_index, vocab_size)

def longest_sentence(X_train, X_val, X_test):
    return max(
        len(sorted(X_train, key=len, reverse=True)[0]),
        len(sorted(X_val, key=len, reverse=True)[0]),
        len(sorted(X_test, key=len, reverse=True)[0]))


def create_embedding_matrix(vocab_size, EMBEDDING_DIM, word_index):
    EMBEDDING_DIM = EMBEDDING_DIM
    # embedding_index will be the dict containing words in word_index, but GloVe coefficients
    embedding_index = {}
    # embedding_matrix dimensions: vocab_size x embedding_dim
    embedding_matrix = np.zeros([vocab_size, EMBEDDING_DIM])
    if (EMBEDDING_DIM == 50):
        f = open(path.join(DATA_DIRECTORY, 'GloVe', 'glove.6B.50d.txt'), encoding='utf-8')
    else:
        f = open(path.join(DATA_DIRECTORY, 'GloVe', 'glove.6B.300d.txt'), encoding='utf-8')
    # Format is: word..value..value.. ..value. We split to make a list, and loading in embedding_index
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()  
    # We iterate on word_index to fill the embedding_matrix with words from word_index
    # and values from embedding_index. If there is no value, it is filled with zeros
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix



def embedding_model(vocab_size, EMBEDDING_DIM, embedding_matrix, maxlen):
    # Transformamos nuestos datos de entrenamiento, val y test a 3 dimensiones, teniendo en cuenta su representación con
    # embedding preentrenado
    embed = Sequential()
    embed.add(Embedding(input_dim=vocab_size,
                        output_dim=EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))
    embed.compile()
    return embed

def autoencoding_model(dim, maxlen, EMBEDDING_DIM, X_train, X_test):
    # Creamos el encoder, la primera parte del autoencoder
    encoder = Sequential()
    encoder.add(Flatten(input_shape=[maxlen,EMBEDDING_DIM]))
    encoder.add(Dense(dim, input_shape=[maxlen*EMBEDDING_DIM]))
    # Creamos el decoder, la segunda parte del autoencoder
    decoder = Sequential()
    decoder.add(Dense(maxlen*EMBEDDING_DIM, input_shape=[dim]))
    decoder.add(Reshape([maxlen, EMBEDDING_DIM]))
    # Compilamos el autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(loss='mse',optimizer='adam')
    # fitting
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5)
    autoencoder.fit(X_train, X_train, epochs=20, validation_data=(X_test, X_test), callbacks=[early_stopping])
    return encoder


def ANN_model(neurons_1, neurons_2, drop, X_train_encoded, X_val_encoded, y_train, y_val):
    # Con los datos encoded, creamos una nueva ANN, que tendrá como entrada dichos datos de entrenamiento
    model = Sequential()
    model.add(Dense(neurons_1, activation='relu', input_shape=[X_train_encoded.shape[1]]))
    model.add(Dropout(drop))
    model.add(Dense(neurons_2, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    history = model.fit(X_train_encoded,
                        y_train,
                        batch_size=64,
                        epochs=300,
                        validation_data=(X_val_encoded, y_val),
                        callbacks=early_stopping)
    return (model, history)

def CNN_model(vocab_size, EMBEDDING_DIM, maxlen, filters, kernel_size, embedding_matrix,
             neurons_1, neurons_2, neurons_3, drop, X_train, X_val, y_train, y_val):
    # Creamos nuestro modelo base con las siguientes capas:
    #    - Embedding: Se encarga de asignar las 50 dimensiones a cada palabra del tweet de entrada (maxlen=92).
    #    - Conv1D: Aplica la convolución en una dimensión, utilizando una región marcada por kernel_size.
    #    - GlobalMaxPooling1D: Hace downsampling de la representación de entrada, tomando el valor máximo del conjunto.
    #    - Dense: Capa oculta que contiene las neuronas para crear la red neuronal artificial.
    #    - Dropout: Capa que desactiva las entradas de la capa de anterior con un ratio definido que ayuda a reducir overfitting.
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))
    model.add(Conv1D(filters=filters,
                    kernel_size=kernel_size,
                    activation='relu'))
    if (neurons_3 != 0):
        model.add(Conv1D(filters=filters/2,
                        kernel_size=kernel_size-2,
                        activation='relu'))
        model.add(Conv1D(filters=filters/4,
                        kernel_size=kernel_size-2,
                        activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(neurons_1, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(neurons_2, activation='relu'))
    model.add(Dropout(drop))
    if (neurons_3 != 0):
        model.add(Dense(neurons_3, activation='relu'))
        model.add(Dropout(drop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train,
                    y_train,
                    batch_size=64,
                    epochs=100,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])
    return (model, history)
    