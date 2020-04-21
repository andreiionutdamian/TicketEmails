# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:04:18 2020

@author: Andrei
"""

import numpy as np
import os
import textwrap 

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



def classify_by_neighbor(tokens, embeds, dct_n2i, thr=0.5):
  idxs = [dct_n2i[w] for w in CLASSES]
  np_class_emb = np.array([embeds[x] for x in idxs])
  _count = np.zeros(len(CLASSES))
  for token in tokens:
    dists = utils.cos_dist(embeds[token], np_class_emb)
    dists_filt = np.where(dists <= thr, dists, np.ones(dists.shape) * np.inf)
    _best = np.argmin(dists_filt)
    if dists_filt[_best] < 2:
      _count[_best] += 1
  return np.argmax(_count)


def load_defs(dct_n2i, dct_i2n):
  data = []
  labels = []
  texts = []
  for label, class_name in enumerate(CLASSES):
    data_folder = os.path.join(DATA_FOLDER, class_name)
    files = os.listdir(data_folder)
    for file in files:
      if '.def' not in file:
        continue
      fn = os.path.join(data_folder, file)
      with open(fn, 'rt', encoding="utf-8") as fh:
        text = fh.read()
      obs = utils.tokenize_as_list(text[1:],  dct_n2i)
      txt = ' '.join([dct_i2n[t] for t in obs])
      texts.append(txt)
      data.append(obs)
      labels.append(label)
  return data, labels, texts

if __name__ == '__main__':
  EMBS_FILE = os.path.join(DATA_FOLDER, 'embs_voc.npz')

  log = utils.Log()

  if 'np_embeds' not in globals():
    log.P("Loading GloVe word embeddings")
    glove_words = os.path.join(EMBS_FILE)
    data = np.load(glove_words)
    np_vocab = data['arr_0']
    np_embeds = data['arr_1']
    dct_i2n = {x:np_vocab[x] for x in range(np_vocab.shape[0])}
    dct_n2i = {np_vocab[x]:x for x in range(np_vocab.shape[0])}
    
  def word_cls(word):
    token = dct_n2i.get(word)
    assert token
    idxs = [dct_n2i[w] for w in CLASSES]
    np_class_emb = np.array([np_embeds[x] for x in idxs])
    dists = utils.cos_dist(np_embeds[token], np_class_emb)
    log.P("Dists for '{}':".format(word))
    for i in range(len(CLASSES)):
      log.P("  {:<13} {:.3f}".format(CLASSES[i] + ':', dists[i]))

  data, labels, texts = load_defs(dct_n2i, dct_i2n)
  
  for i,txt in enumerate(texts):
    log.P("Text '{}':\n{}".format(
        CLASSES[i],
        textwrap.indent(textwrap.fill(txt),'  '),
        ))
    yp = classify_by_neighbor(data[i], np_embeds, dct_n2i)
    log.P("Predicted: {}\n".format(CLASSES[yp]))