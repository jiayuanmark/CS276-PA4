import sys
import re
from math import log10, log, exp
import pickle
from sklearn import preprocessing

# global parameters for BM25F
B = {'title':0.3, 'header':0.5, 'url':2.0, 'body':1.0, 'anchor':0.1}
W = {'title':1.0, 'header':0.5, 'url':0.1, 'body':0.3, 'anchor':2.0}
K = 0.8

# vector operations
def vector_from_text(items, content):
    vec = [0] * len(items)
    content = content.split()
    for idx in range(len(items)):
        vec[idx] = content.count(items[idx])
    return vec

def vector_product(vec1, vec2):
    assert(len(vec1) == len(vec2))
    return [ (vec1[i] * vec2[i]) for i in range(len(vec1)) ]

def vector_dot_product(vec1, vec2):
    return sum(vector_product(vec1, vec2))

def vector_diff(vec1, vec2):
    assert(len(vec1) == len(vec2))
    return [ (vec1[i] - vec2[i]) for i in range(len(vec1)) ]

def vector_sum(vec1, vec2):
    assert(len(vec1) == len(vec2))
    return [ (vec1[i] + vec2[i]) for i in range(len(vec1)) ]

def vector_scale(vec, alpha):
    return [ (u * alpha) for u in vec ]

def sublinear_scale(vec):
    rvec = []
    for u in vec:
        if u == 0:
            rvec.append(0)
        else:
            rvec.append(1 + log(u))
    return rvec

# return inverse document frequencies for terms
def document_frequency(items, table):
    vec = [0] * len(items)
    for idx in range(len(items)):
        vec[idx] = log10(table[items[idx]])
    return vec

# load document frequency table
def load_doc_freq():
    table = {}
    with open('term_doc_freq', 'rb') as ff:
        table = pickle.load(ff)
    return table


def read_feature_file(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    documents = {}
    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        documents[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        documents[query][url] = {}
      elif(key == 'title'):
        documents[query][url][key] = value
      elif(key == 'header'):
        curHeader = documents[query][url].setdefault(key, [])
        curHeader.append(value)
        documents[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in documents[query][url]:
          documents[query][url][key] = {}
        temp = value.split(' ', 1)
        documents[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        documents[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in documents[query][url]:
          documents[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        documents[query][url]['anchors'][anchor_text] = int(value)
    f.close()
    return (queries, documents)


def build_features(queries, documents):
  features, qryDocList = [], []
  table = load_doc_freq()

  for query in queries.keys():
    # Query item and query vector
    qitem = list(set(query.split()))
    qvec = sublinear_scale(vector_from_text(qitem, query))
    idf = document_frequency(qitem, table)
    qvec = vector_product(qvec, idf)
        
    # Calculate tf-idf score
    results = queries[query]
    for x in results:
      feat, fld_len = [], []
      # title
      title = documents[query][x]['title']
      title_vec = vector_from_text(qitem, title)
      feat.append(vector_dot_product(qvec, title_vec))
      fld_len.append(len(title.split()))

      # url
      url = re.sub(r'\W+', ' ', x)
      url_vec = vector_from_text(qitem, url)
      feat.append(vector_dot_product(qvec, url_vec))
      fld_len.append(len(url.split()))

      # header
      val = 0
      header_len = 0
      if 'header' in documents[query][x]:
        header_arr = documents[query][x]['header']
        for header in header_arr:
          val = val + vector_dot_product(qvec, vector_from_text(qitem, header))
          header_len += len(header.split())
      feat.append(val)
      fld_len.append(header_len)

      # body
      if 'body_hits' in documents[query][x]:
        body = documents[query][x]['body_hits']
        body_vec = [len(body.setdefault(item, [])) for item in qitem]
        feat.append(vector_dot_product(qvec, body_vec))
      else:
        feat.append(0)
      fld_len.append(int(documents[query][x]['body_length']) + 500)

      # achors
      val = 0
      anchor_len = 0
      if 'anchors' in documents[query][x]:
        anchor = documents[query][x]['anchors']
        for key in anchor:
          val = val + vector_dot_product(qvec, [anchor[key] * u for u in vector_from_text(qitem, key)])
          anchor_len += len(key.split())
      feat.append(val)
      fld_len.append(anchor_len)

      # length normalization
      norm = int(documents[query][x]['body_length']) + 500
      #feat = [float(u) / float(norm) for u in feat]
      for i in range(len(feat)):
        if feat[i] != 0:
          feat[i] /= fld_len[i]


      features.append(feat)
      qryDocList.append((query, x))
  return (features, qryDocList)


def avg_field_len(documents):
    title, header, body, url, anchor = [], [], [], [], []
    for query in documents:
        for x in documents[query]:
            title.append(len(documents[query][x]['title'].split()))
            url.append(len(re.sub(r'\W+', ' ', x).split()))
            body.append(int(documents[query][x]['body_length'])+500)
            if 'header' in documents[query][x]:
                hlen = 0
                for h in documents[query][x]['header']:
                    hlen = hlen + len(h.split())
                header.append(hlen)
            if 'anchors' in documents[query][x]:
                anchor_arr = documents[query][x]['anchors']
                alen = 0
                for key in anchor_arr:
                    alen = alen + len(key.split()) * anchor_arr[key]
                anchor.append(alen)
    avglen = {}
    avglen['title'] = sum(title) / float(len(title))
    avglen['header'] = sum(header) / float(len(header))
    avglen['body'] = sum(body) / float(len(body))
    avglen['url'] = sum(url) / float(len(url))
    avglen['anchor'] = sum(anchor) / float(len(anchor))
    return avglen


def BM25F_score(qvec, term_vec, fld_len, avg_len):
  global B, W, K
  vec = [0] * len(qvec)
  for field in term_vec:
    vec = vector_sum(vec, vector_scale(sublinear_scale(term_vec[field]), \
                          W[field] / (1.0 + B[field] * (fld_len[field] / float(avg_len[field]) - 1.0))))
  score = 0.0
  for id in range(len(qvec)):
    score = score + vec[id] / (vec[id] + K) * qvec[id]
  return score


def compute_window(qitem, text):
  if not set(text).issuperset(set(qitem)):
    return sys.maxint
  if len(qitem) == 1:
    return 1
  index = [text.index(u) for u in qitem]
  win = max(index) - min(index) + 1
  for i in range(len(text)):
    word = text[i]
    if word in qitem:
      index[qitem.index(word)] = i
      if max(index) - min(index) + 1 < win:
        win = max(index) - min(index) + 1
  return win


def compute_body_window(qitem, body):
  if not set(body.keys()).issuperset(set(qitem)):
    return sys.maxint
  if len(qitem) == 1:
    return 1
  lst = []
  for id in range(len(qitem)):
    for idx in body[qitem[id]]:
      lst.append((idx, id))
  lst = sorted(lst, key=lambda x: x[0], reverse=False)

  win = sys.maxint
  count = [0] * len(qitem)
  left, right, zeros = 0, 0, len(qitem)
  count[lst[left][1]] += 1
  zeros -= 1
  while right < len(lst):
    if zeros != 0:
      right += 1
      if right == len(lst):
        break
      if count[lst[right][1]] == 0:
        zeros -= 1
      count[lst[right][1]] += 1
    else:
      if lst[right][0] - lst[left][0] + 1 < win:
        win = lst[right][0] - lst[left][0] + 1
      count[lst[left][1]] -= 1
      if count[lst[left][1]] == 0:
        zeros += 1
      left += 1
  return win


def build_rich_features(queries, documents):
  features, qryDocList = [], []
  table = load_doc_freq()
  avglen = avg_field_len(documents)

  for query in queries.keys():
    # Query item and query vector
    qitem = list(set(query.split()))
    qvec = sublinear_scale(vector_from_text(qitem, query))
    #qvec = vector_from_text(qitem, query)
    idf = document_frequency(qitem, table)
    qvec = vector_product(qvec, idf)
        
    # Calculate tf-idf score
    results = queries[query]
    for x in results:
      feat = []
      term_vec, fld_len = {}, {}

      # title
      title = documents[query][x]['title']
      title_vec = vector_from_text(qitem, title)
      term_vec['title'] = title_vec
      fld_len['title'] = len(title.split())
      
      # url
      url = re.sub(r'\W+', ' ', x)
      url_vec = vector_from_text(qitem, url)
      term_vec['url'] = url_vec
      fld_len['url'] = len(url.split())

      # header
      header_vec = [0] * len(qitem)
      header_len = 0
      if 'header' in documents[query][x]:
        for header in documents[query][x]['header']:
          header_vec = vector_sum(header_vec, vector_from_text(qitem, header))
          header_len = header_len + len(header.split())
      term_vec['header'] = header_vec
      fld_len['header'] = header_len

      # body
      body_vec = [0] * len(qitem)
      if 'body_hits' in documents[query][x]:
        body = documents[query][x]['body_hits']
        body_vec = [len(body.setdefault(item, [])) for item in qitem]
      term_vec['body'] = body_vec
      fld_len['body'] = int(documents[query][x]['body_length']) + 500

      # achors
      anchor_vec = [0] * len(qitem)
      anchor_len = 0
      if 'anchors' in documents[query][x]:
        anchor = documents[query][x]['anchors']
        for key in anchor:
          anchor_vec = vector_sum(anchor_vec, [anchor[key] * u for u in vector_from_text(qitem, key)])
          anchor_len = anchor_len + len(key.split()) * anchor[key]
      term_vec['anchor'] = anchor_vec
      fld_len['anchor'] = anchor_len

      # length normalization using field length
      for key in term_vec:
        val = vector_dot_product(qvec, term_vec[key])
        if val != 0:
          val /= fld_len[key]
        feat.append(val)


      # url ends in PDF
      if x.endswith('.pdf'):
        feat.append(1)
      else:
        feat.append(0)
      

      # BM25F
      feat.append(BM25F_score(qvec, term_vec, fld_len, avglen))
      
      # title window
      feat.append(compute_window(qitem, title.split()))
         
      # url window
      feat.append(compute_window(qitem, url.split()))
      
      # body window
      '''
      if 'body_hits' in documents[query][x]:
        feat.append(compute_body_window(qitem, documents[query][x]['body_hits']))
      else:
        feat.append(sys.maxint)
      '''

      # header window
      header_win = [sys.maxint]
      if 'header' in documents[query][x]:
        header_win.extend([ compute_window(qitem, u.split()) for u in documents[query][x]['header'] ])
      feat.append( min(header_win) )
      
      # anchor window
      anchor_win = [sys.maxint]
      if 'anchors' in documents[query][x]:
        anchor_win.extend([ compute_window(qitem, key.split()) for key in documents[query][x]['anchors'] ])
      feat.append( min(anchor_win) )
      
      # page rank
      feat.append(documents[query][x]['pagerank'])

      features.append(feat)
      qryDocList.append((query, x))
  return (features, qryDocList)


    

def build_labels(labelFile, qryDocList):
  labels = [0] * len(qryDocList)
  with open(labelFile) as f:
    for line in f.readlines():
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if key == 'query':
        query = value
      elif key == 'url':
        url = value.split()[0].strip()
        score = float(value.split()[1].strip())
        labels[qryDocList.index((query, url))] = score
  return labels


def build_indexmap(qryDocList):
  queries = []
  index_map = {}
  for i in range(len(qryDocList)):
    (qry, url) = qryDocList[i]
    if qry not in index_map:
      queries.append(qry)
      index_map[qry] = {}
    else:
      index_map[qry][url] = i
  return queries, index_map


def build_pair_data(queries, features, qryDocList, labels):
  features = preprocessing.scale(features)
  X, y = [], []
  for qry in queries:
    urls = queries[qry]
    for i in range(len(urls)):
      id1 = qryDocList.index((qry, urls[i]))
      for j in range(i+1, len(urls)):
        id2 = qryDocList.index((qry, urls[j]))
        X.append(vector_diff(features[id1], features[id2]))
        if labels[id1] >= labels[id2]:
          y.append(1)
        else:
          y.append(-1)
  return (X, y)


def print_results(queries, index_map, y):
    for query in queries:
      print("query: " + query)
      res = {}
      for url in index_map[query]:
        res[url] = y[index_map[query][url]]
      res = sorted(res.items(), key=lambda x: x[1], reverse=True)
      for item in res:
        print("  url: " + item[0])

def print_to_file_results(queries, index_map, y):
    with open("ranked.txt", 'w') as ff:
      for query in queries:
        ff.write("query: " + query + "\n")
        res = {}
        for url in index_map[query]:
          res[url] = y[index_map[query][url]]
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        for item in res:
          ff.write("  url: " + item[0] + "\n")