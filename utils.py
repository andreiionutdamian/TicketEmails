# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:02:18 2020

@author: Andrei
"""
import numpy as np
from datetime import datetime as dt
import pandas as pd


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

def cos_dist(v, embeds):
  dists = np.maximum(0, 1 - embeds.dot(v) / (np.linalg.norm(v) * np.linalg.norm(embeds, axis=1)))
  return dists

def neighbors_by_idx(idx, embeds, k=None):
  v = embeds[idx]
  dists = cos_dist(v, embeds)  
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


def tokenize_and_embeds(text, embeds, dct_n2i, max_size=500):  
  idxs = tokenize_as_list(text, dct_n2i=dct_n2i)
  zeros = np.zeros(embeds.shape[1])
  out_embeds = np.array([embeds[x] for x in idxs]).astype(np.float32)
  if max_size is not None:
    nr_pad = max(0, max_size - out_embeds.shape[0])
    if nr_pad > 0:
      np_pad = np.array([zeros for _ in range(nr_pad)])
      out_embeds = np.concatenate(
          (out_embeds, np_pad))
    out_embeds = out_embeds[:max_size]
    assert out_embeds.shape == (max_size,embeds.shape[1])
  return out_embeds


def tokenize_as_list(text, dct_n2i):
  splitted = [w.lower() for w in text.split()]
  idxs = [dct_n2i.get(w) for w in splitted]
  idxs = [x for x in idxs if x != None]
  return idxs