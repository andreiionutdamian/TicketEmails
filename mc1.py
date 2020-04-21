# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:46:27 2020

@author: damian

TODO:
  Classes:
    1. "Completeness"
    2. "Consistency"
    3. "Conformity"
    4. "Accuracy"
    5. "Integrity"
    6. "Timeliness"

"""
import numpy as np
import pandas as pd
import os


import tensorflow as tf

import utils

CLASSES = [
      "completeness",
      "consistency",
      "conformity",
      "accuracy",
      "integrity",
      "timeliness",
      ]

DATA_FOLDER = 'data'


def load_train_data(embeds, dct_n2i, seq_size=500):
  data = []
  labels = []
  for label, class_name in enumerate(CLASSES):
    data_folder = os.path.join(DATA_FOLDER, class_name)
    files = os.listdir(data_folder)
    for file in files:
      fn = os.path.join(data_folder, file)
      with open(fn, 'rt', encoding="utf-8") as fh:
        text = fh.read()
      obs = utils.tokenize_and_embeds(text, embeds, dct_n2i, max_size=seq_size)
      data.append(obs)
      labels.append(label)
  return np.array(data), np.array(labels)     




def decode(obs, embeds, dct_i2n):
  zeros = np.zeros(obs.shape[1])
  valid = [x for x in obs if np.all(x != zeros)]
  idxs = []
  for embed in valid:
    diff = np.abs((embeds  - embed)).sum(axis=1)
    idx = np.argmin(diff)
    idxs.append(idx)
  texts = [dct_i2n[i] for i in idxs]
  txt = " ".join(texts)
  return txt


def get_test_data(df, embeds, dct_n2i, seq_size=500):
  texts = [x for x in df.Description]
  data = [utils.tokenize_and_embeds(doc, embeds=embeds, dct_n2i=dct_n2i, max_size=seq_size) 
          for doc in texts]
  return np.array(data)
    

def get_model(input_shape):
  tf_input = tf.keras.layers.Input(input_shape)
  tf_x = tf_input
  tf_x1 = tf.keras.layers.Conv1D(256, 1, activation='relu')(tf_x)
  tf_x2 = tf.keras.layers.Conv1D(256, 3, activation='relu')(tf_x)
  tf_x3 = tf.keras.layers.Conv1D(256, 5, activation='relu')(tf_x)
  
  tf_x1 = tf.keras.layers.LSTM(256)(tf_x1)
  tf_x2 = tf.keras.layers.LSTM(256)(tf_x2)
  tf_x3 = tf.keras.layers.LSTM(256)(tf_x3)
  
  tf_x = tf.keras.layers.concatenate([tf_x1, tf_x2, tf_x3])
  tf_x = tf.keras.layers.Dropout(0.7)(tf_x)
  tf_x = tf.keras.layers.Dense(384, activation='relu')(tf_x)  
  tf_x = tf.keras.layers.Dropout(0.7)(tf_x)
  
  tf_out = tf.keras.layers.Dense(len(CLASSES), 
                                 activation='softmax')(tf_x)
  
  model = tf.keras.models.Model(tf_input, tf_out, name='MC')
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer='nadam',
      metrics=['acc']
      )
  return model

def save_data(df, log):
  df_hl = df[~df.DQType.isna()]
  log.P("Found {} labels".format(df_hl.shape[0]))
  for i in range(df_hl.shape[0]):
    subfolder = df_hl.iloc[i].DQType
    file = df_hl.iloc[i].Key
    fn = os.path.join(DATA_FOLDER, subfolder, file.lower() + '.txt')
    txt = df_hl[df_hl.Key == file][['Description']].iloc[0,0]
    with open(fn, 'wt', encoding="utf-8") as fh:
      fh.write(txt)
  return
  

if __name__ == '__main__':
  
  SEQ_SIZE = 100
  
  FULL_TRAIN = True
  
  
  GLV_FILE = os.path.join(DATA_FOLDER, 'glove.6B.50d.txt')
  EMBS_FILE = os.path.join(DATA_FOLDER, 'embs_voc.npz')
  DATA_FILE = os.path.join(DATA_FOLDER, 'data3.xlsx')
  log = utils.Log()
  

  
  for c in CLASSES:
    _dir = os.path.join(DATA_FOLDER, c)
    if not os.path.isdir(_dir):
      os.mkdir(_dir)
  df_inp = pd.read_excel(DATA_FILE)
  df = df_inp[~df_inp.Description.isna()]
  save_data(df, log=log)
  
  if 'np_embeds' not in globals():
    if os.path.isfile(EMBS_FILE):
      log.P("Loading GloVe word embeddings")
      glove_words = os.path.join(EMBS_FILE)
      data = np.load(glove_words)
      np_vocab = data['arr_0']
      np_embeds = data['arr_1']
      dct_i2n = {x:np_vocab[x] for x in range(np_vocab.shape[0])}
      dct_n2i = {np_vocab[x]:x for x in range(np_vocab.shape[0])}
    else:
      log.P("Generating word vectors...")
      _d = utils.glove2dict(GLV_FILE)
      word_list = list(_d.keys())
      np_vocab = np.array(word_list)
      np_embeds = np.array(
          [_d[np_vocab[x]] 
          for x in range(np_vocab.shape[0])]
          ).astype(np.float32)
      log.P("Word vectors generated.")
      log.P("Saving word vectors...")
      np.savez(EMBS_FILE, np_vocab, np_embeds)
      log.P("Saved embeds.")
      dct_i2n = {x:np_vocab[x] for x in range(np_vocab.shape[0])}
      dct_n2i = {np_vocab[x]:x for x in range(np_vocab.shape[0])}      
  else:
    log.P("np_embeds already loaded.")
      
  def show_word(word):
    return utils.show_neighbors(
        idx=word, 
        embeds=np_embeds, 
        dct_i2n=dct_i2n, 
        log=log,
        dct_n2i=dct_n2i)
  
  for c in CLASSES:
    show_word(c)
    
  if FULL_TRAIN:
    log.P("Prepare data...")
    X, y = load_train_data(
        embeds=np_embeds,
        dct_n2i=dct_n2i,
        seq_size=SEQ_SIZE)
    log.P("Done prepare data.")
    
    if False:
      log.P(decode(X[0], np_embeds, dct_i2n))
    
    model = get_model(X.shape[1:])
    model.fit(X, y, epochs=100)
    
    
    x_test = get_test_data(df, np_embeds, dct_n2i, seq_size=SEQ_SIZE)
    
    yh = model.predict(x_test)
    yp = yh.argmax(axis=1)
    labels = [CLASSES[y] for y in yp]
    procs = [round(p[x] * 100,2) for p, x in zip(yh, yp)]
    df_res = pd.DataFrame({
        'KEY' : df.Key,
        'LABEL' : labels,
        'PROBA' : procs
        })
    df_res.to_csv('results.csv', index=False)
    
  
  
  