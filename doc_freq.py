import os
import pickle

table = {}
with open('df.txt', 'r') as ff:
    for line in ff.readlines():
        key = line.split()[0].strip()
        val = int(line.split()[1].strip())
        table[key] = val


dic = {}
with open('term_doc_freq', 'rb') as f:
    dic = pickle.load(f)

print dic['selfop']
