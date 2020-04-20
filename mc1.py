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
from datetime import datetime as dt

import tensorflow as tf

CLASSES = [
      "completeness",
      "consistency",
      "conformity",
      "accuracy",
      "integrity",
      "timeliness",
      ]

DATA_FOLDER = 'data'

class Log():
  def __init__(self):      
    self.lst_log = []
    self._date = dt.now().strftime("%Y%m%d_%H%M")
    self.log_fn = dt.now().strftime("logs/"+self._date+"_log.txt")
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('precision', 4)  
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=500)
#    plt.style.use('ggplot')
    return

  def P(self, s=''):
    self.lst_log.append(s)
    print(s, flush=True)
    try:
      with open(self.log_fn, 'w') as f:
          for item in self.lst_log:
              f.write("{}\n".format(item))
    except:
      pass
    return
  
  def Pr(self, s=''):
      print('\r' + str(s), end='', flush=True)

def neighbors_by_idx(idx, embeds, k=None):
  v = embeds[idx]
  dists = np.maximum(0, 1 - embeds.dot(v) / (np.linalg.norm(v) * np.linalg.norm(embeds, axis=1)))
  idxs = np.argsort(dists)
  return idxs[:k], dists[idxs][:k]

def show_neighbors(idx, 
                   embeds, 
                   dct_i2n, 
                   dct_n2i,
                   log,
                   k=10,):
  if type(idx) != int:
    if type(idx) == str:
      idx = idx.lower()
    idx = dct_n2i[idx]
  idxs, dists = neighbors_by_idx(idx, embeds, k=k)
  max_len = max([len(str(dct_i2n[ii])) for ii in idxs]) + 1
  log.P("Top neighbors for '{}'".format(dct_i2n[idx]))
  for i,ii in enumerate(idxs):
    log.P(("  {:<" + str(max_len) + "} {:.3f}").format(str(dct_i2n[ii]) + ':', dists[i]))
  
  

def glove2dict(src_filename):
  data = {}
  with open(src_filename, encoding='utf8') as f:
    while True:
      try:
        line = next(f)
        line = line.strip().split()
        data[line[0]] = np.array(line[1: ], dtype=np.float)
      except StopIteration:
        break
      except UnicodeDecodeError:
        pass
  return data


def tokenize(text, embeds, dct_n2i, max_size=500):
  splitted = [w.lower() for w in text.split()]
  idxs = [dct_n2i.get(w) for w in splitted]
  zeros = np.zeros(embeds.shape[1])
  embeds = np.array([embeds[x] for x in idxs if x != None]).astype(np.float32)
  nr_pad = max(0, max_size - embeds.shape[0])
  embeds = np.concatenate(
      (embeds, [zeros for _ in range(nr_pad)]))
  return embeds


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
      obs = tokenize(text, embeds, dct_n2i, max_size=seq_size)
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


def get_test_data(df, embeds, dct_n2i):
  texts = [x for x in df.Description.apply(lambda x: x.split())]
  data = [tokenize(doc) for doc in texts]
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
  tf_x = tf.keras.layers.Dense(384, activation='relu')(tf_x)  
  
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
  
  FULL_TRAIN = True
  
  GLV_FILE = os.path.join(DATA_FOLDER, 'glove.6B.50d.txt')
  EMBS_FILE = os.path.join(DATA_FOLDER, 'embs_voc.npz')
  DATA_FILE = os.path.join(DATA_FOLDER, 'data2.xlsx')
  log = Log()
  

  
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
      _d = glove2dict(GLV_FILE)
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
    return show_neighbors(
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
        seq_size=500)
    log.P("Done prepare data.")
    
    log.P(decode(X[0], np_embeds, dct_i2n))
    
    model = get_model(X.shape[1:])
    model.fit(X, y, epochs=10)
    
    
    x_test = get_test_data(df, np_embeds, dct_n2i)
    
    yh = model.predict(x_test).argmax(axis=1)
    labels = [CLASSES[y] for y in yh]
    df_res = pd.DataFrame({
        'KEY' : df.Key,
        'LABEl' : labels
        })
    df_res.to_csv('results.csv')
    
  
  
  