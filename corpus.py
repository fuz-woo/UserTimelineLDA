#import jieba
import os

def cut_corpus(dn,of):
    fl = [dn+'/'+fn for fn in os.listdir(dn)]
    writer = open(of,'a')
    for fn in fl:
        print "handling file",fn,"..."
        fd = open(fn)
        for line in fd:
            k,v = line.decode("utf-8").split("\t\t")
            ws = [w for w in jieba.cut(k)]
            writer.write(" ".join(ws).encode("utf-8")+"\t\t"+v.encode("utf-8"))
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
