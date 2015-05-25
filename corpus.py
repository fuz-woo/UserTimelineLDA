import jieba
import os
import cPickle as pickle

stopwords = pickle.load(open("corpus/stop_words.dump"))

def cut_corpus(fn,of):
    writer = open(of,'a')
    fd = open(fn)
    for line in fd:
        ws,ul = line.decode("utf-8").strip().split("\t")
        ws = [w for w in jieba.cut(ws) if len(w.encode('utf-8'))>1 and not w.isdigit() and not w.lower().islower()]
        if len(ws) > 30:
            writer.write((" ".join(ws)+"\t\t"+ul+'\n').encode('utf-8'))
    fd.close()
    writer.close()

def load_corpus(fn):
    corpus = []
    fd = open(fn)
    for line in fd:
        k,v = line.strip().decode("utf-8").split("\t\t")
        doc,ul = (k.strip().split(' '),map(lambda s:s.split(","),v.strip().split(' ')))
        corpus.append((doc,[(float(t),u) for t,u in ul]))
    fd.close()
    return corpus

def length_limit(doc,ul,doc_len=40,ul_len=40):
    return len(doc) >= doc_len and len(ul) >= ul_len

import time
monday = time.mktime((2014,7,1,0,0,0,0,0,0))
sunday = time.mktime((2014,7,8,0,0,0,0,0,0))
dt = sunday - monday
def format_date(t):
    return (t - monday) / dt

def first_week(t,u):
    return t < sunday

def format_date_daily(t):
    return t%86400 / 86400.



def format_corpus(fn,limit=length_limit,filter_doc=None,filter_ul=first_week,format_date=format_date,stop_words=set([])):
    corpus = []
    fd = open(fn)
    i = 0
    for line in fd:
        doc,ul = line.strip().decode("utf-8").split("\t\t")
        doc = [w for w in doc.split(" ") if w not in stop_words]
        ul = map(lambda s:s.split(","),ul.split(" "))
        ul = [(float(t)/1000,u) for t,u in ul]
        if format_date:
            try:
                ul = [(format_date(t),long(u)) for t,u in ul if filter_ul(t,u)]
                if limit is None or limit(doc,ul):
                    print i
                    i+=1
                    corpus.append((doc,ul))
            except Exception as e:
                print e.message
    return corpus


def proc_data(fn):
    fd = open(fn)
    ds = [line.decode("utf-8").strip().split("\t") for line in fd]
    corpus = []
    for dw,ul in ds:
        doc = [w for w in jieba.cut(dw) if len(w.encode('utf-8'))>1 and not w.isdigit() and not w.lower().islower()]
        ul = map(lambda s:s.split(","),ul.split(" "))
        corpus.append((doc,ul))
    return corpus
