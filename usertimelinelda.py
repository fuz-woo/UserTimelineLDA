# -*- coding: utf-8 -*-

import os
import random
import math
from numpy import random as nprand
import copy as cp
import corpus as cputil
import beta
import cPickle as pickle

#corpus = cputil.load_corpus("corpus/corpus-week-normalized.csv")

#corpus = [([w for w in doc if len(w)>1 and not w.isdigit() and not w.lower().islower()],ul) for doc,ul in corpus]
single_step = 1. / 86400.
corpus = pickle.load(open("corpus/corpus-user-freq-filtered.dump"))


class UserTimelineLDA:
    def __init__(self,corpus,alpha=0.1,beta=0.01,gamma=0.01,K=20,n_iter=300):
        self.corpus = corpus
        self.corpus_word = [w for w,_ in self.corpus]
#        self.corpus_word,users = zip(*self.corpus)
        self.corpus_user = [[u for _,u in ul] for _,ul in self.corpus]
#        self.corpus_user,self.corpus_timestamp = zip(*users)
        self.corpus_timestamp = [[t for t,_ in ul] for _,ul in self.corpus]
        self.M = len(self.corpus)
        self.NW = sum(map(len,self.corpus_word))
        self.NU = sum(map(len,self.corpus_user))
        
        self.wdic = list(set([w for d in self.corpus_word for w in d]))
        self.wsize = len(self.wdic)
        self.udic = list(set([u for d in self.corpus_user for u in d]))
        self.usize = len(self.udic)
        self.wmap = {w:i for i,w in enumerate(self.wdic)}
        self.umap = {u:i for i,u in enumerate(self.udic)}
        self.corpus_word = [[self.wmap[w] for w in wl] for wl in self.corpus_word]
        self.corpus_user = [[self.umap[u] for u in ul] for ul in self.corpus_user]

        self.K = K
        self.C = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_iter = n_iter
        self.topics = [nprand.randint(0,self.K,size=l) for l in map(len,self.corpus_word)]
        self.communities = [nprand.randint(0,self.K,size=l) for l in map(len,self.corpus_user)]

        self.Kalpha = self.K * self.alpha
        self.wbeta = self.wsize * self.beta
        self.ugamma = self.usize * self.gamma


        print "corpus: ",self.M
        print "word count: ",self.wsize
        print "user count: ",self.usize

        self.n_m = [[0 for k in xrange(self.K)] for m in xrange(self.M)]
        self.n_m_sum = [0 for m in xrange(self.M)]
#        self.n_z = [{w:0 for w in self.wdic} for k in xrange(self.K)]
        self.n_z = [[0 for w in xrange(self.wsize)] for k in xrange(self.K)]
        self.n_z_sum = [0 for k in xrange(self.K)]

        self.n_t = [[0 for c in xrange(self.C)] for m in xrange(self.M)]
        self.n_t_sum = [0 for m in xrange(self.M)]
#        self.n_c = [{u:0 for u in self.udic} for c in xrange(self.C)]
        self.n_c = [[0 for u in xrange(self.usize)] for c in xrange(self.C)]
        self.n_c_sum = [0 for c in xrange(self.C)]
        self.ts = [[] for c in xrange(self.C)]
        self.ts_sum = [0. for c in xrange(self.C)]

        for m in xrange(self.M):
            doc_word = self.corpus_word[m]
            doc_user = self.corpus_user[m]
            for i in xrange(len(doc_word)):
                w = self.corpus_word[m][i]
                z = self.topics[m][i]
                self.n_m[m][z] += 1
                self.n_z[z][w] += 1
                self.n_m_sum[m] += 1
                self.n_z_sum[z] += 1

            for i in xrange(len(doc_user)):
                u = self.corpus_user[m][i]
                ts = self.corpus_timestamp[m][i]
                c = self.communities[m][i]
                self.n_t[m][c] += 1
                self.n_c[c][u] += 1
                self.n_t_sum[m] += 1
                self.n_c_sum[c] += 1

        self.theta = [[0 for k in xrange(self.K)] for m in xrange(self.M)]
#        self.phi = [{w:0 for w in self.wdic} for k in xrange(self.K)]
        self.phi = [[0 for w in xrange(self.wsize)] for k in xrange(self.K)]
#        self.rho = [{u:0 for u in self.udic} for c in xrange(self.C)]
        self.rho = [[0 for u in xrange(self.usize)] for c in xrange(self.C)]
        self.psi = [(0.8,2) for c in xrange(self.C)]
        self.nstats = 0

        self.perplexities = []


    def run(self,n_iter=300):
        self.n_iter = n_iter
        for n in xrange(self.n_iter):
            print "iteration ",n
            self.gibbs_routine()
            self.update_params()
            pplex = self.perplexity()
            print "perplexity: ",pplex
            self.perplexities.append(pplex)

    def gibbs_routine(self):
        for m in xrange(self.M):
            print "sampling document %s/%s ..." % (m,self.M)
            dw = self.corpus_word[m]
            du = self.corpus_user[m]

            for wi in xrange(len(dw)):
                #print "sampling topics %s/%s,%s/%s " % (m,len(self.corpus),i,len(d))
                z = self.topics[m][wi]
                w = self.corpus_word[m][wi]
                # update document & topic count
                self.n_m[m][z] -= 1
                self.n_z[z][w] -= 1
                # update document & topic sum
                self.n_m_sum[m] -= 1
                self.n_z_sum[z] -= 1
                
                self.topics[m][wi] = self.sample_topic(m,wi)
                newz = self.topics[m][wi]
                
                self.n_m[m][newz] += 1
                self.n_z[newz][w] += 1
                self.n_m_sum[m] += 1
                self.n_z_sum[newz] += 1
            
            for ui in xrange(len(du)):
                #print "sampling communities %s/%s , %s/%s" % (m,len(self.corpus),i,len(du))
                c = self.communities[m][ui]
                u = self.corpus_user[m][ui]
                ts = self.corpus_timestamp[m][ui]
                self.n_t[m][c] -= 1
                self.n_c[c][u] -= 1
                self.n_t_sum[m] -= 1
                self.n_c_sum[c] -= 1
                
                self.communities[m][ui] = self.sample_community(m,ui)
                newc = self.communities[m][ui]
                
                self.n_t[m][newc] += 1
                self.n_c[newc][u] += 1            
                self.n_t_sum[m] += 1
                self.n_c_sum[newc] += 1
                self.ts[newc].append(ts)
                self.ts_sum[newc] += ts


    def sample_topic(self,m,wi):
        w = self.corpus_word[m][wi]
        
#        msum = self.n_m_sum[m] + self.n_t_sum[m] + self.Kalpha
        pz = [( self.n_m[m][k] + self.n_t[m][k] + self.alpha ) * ( self.n_z[k][w] + self.beta ) / (self.n_z_sum[k] + self.wbeta) for k in xrange(self.K)]

        _sum = sum(pz)
        pz = [pi/_sum for pi in pz]

        sample = self.choice(pz)
        return sample
        
    def sample_community(self,t,ui):
        u = self.corpus_user[t][ui]
        ts = self.corpus_timestamp[t][ui]
        
#        msum = self.n_t_sum[t] + self.n_m_sum[t] + self.C * self.alpha
        pc = [self.betap(ts,self.psi[c]) * ( self.n_t[t][c] + self.n_m[t][c] + self.alpha ) * ( self.n_c[c][u] + self.gamma ) / (self.n_c_sum[c] + self.ugamma) for c in xrange(self.C)]
                
        _sum = sum(pc)
        pc = [pi/_sum for pi in pc]

        sample = self.choice(pc)
        return sample

    def update_params(self):
#        print "updating parameters..."
        self.theta = [[(self.n_m[m][k] + self.n_t[m][k] + self.alpha) / (self.n_m_sum[m] + self.n_t_sum[m] + self.Kalpha) for k in xrange(self.K)] for m in xrange(self.M)]
#        self.phi = [{w:(self.n_z[k][w] + self.beta) / (self.n_z_sum[k] + self.wbeta) for w in self.wdic} for k in xrange(self.K)]
        self.phi = [[(self.n_z[k][w] + self.beta) / (self.n_z_sum[k] + self.wbeta) for w in xrange(self.wsize)] for k in xrange(self.K)]
#        self.rho = [{u:(self.n_c[c][u] + self.gamma) / (self.n_c_sum[c] + self.ugamma) for u in self.udic} for c in xrange(self.C)]
        self.rho = [[(self.n_c[c][u] + self.gamma) / (self.n_c_sum[c] + self.ugamma) for u in xrange(self.usize)] for c in xrange(self.C)]

        for c in xrange(self.C):
            _t = self.ts_sum[c] / self.n_c_sum[c]
            _var = sum([(ts-_t)**2 for ts in self.ts[c]]) / self.n_c_sum[c]
            _m = ( _t * (1 - _t) / _var - 1 )
            _a = _t * _m
            _b = (1-_t) * _m
            if _a+_b > 160:
                _a,_b = 160*_a/(_a+_b),160*_b/(_a+_b)
            self.psi[c] = (_a,_b)
            self.ts[c] = []
            self.ts_sum[c] = 0.
            print c,_t,_var,_a,_b
        self.nstats += 1

    def show_topics(self,num_topics=20,num_words=20):
        topics = [sorted(enumerate(phi),key=lambda (w,c):c, reverse=True) for phi in self.phi]
        topics = [topic[:num_words] for topic in topics][:num_topics]
        return topics

    def show_communities(self,num_communities=20,num_words=20):
        communities = [sorted(enumerate(rho),key=lambda (w,c):c, reverse=True) for rho in self.rho]
        communities = [community[:num_words] for community in communities][:num_communities]
        return communities

    def get_phi(self):
        phis = [sorted(enumerate(phi),key=lambda (w,c):c, reverse=True) for phi in self.phi]
        phis = [dict(phi) for phi in phis]
        return phis

    def get_theta(self):
#        thetas = [[kc for kc in doc] for doc in self.theta]
        return self.theta

    def get_rho(self):
        rhos = [sorted(enumerate(rho),key=lambda (w,c):c, reverse=True) for rho in self.rho]
        rhos = [dict(rho) for rho in rhos]
        return rhos

    def perplexity(self):
        phis = self.phi
        thetas = self.theta
        rhos = self.rho

        pplex = 0.
        for m in xrange(self.M):
            doc = self.corpus_user[m]
            for n in xrange(len(doc)):
                u = self.corpus_user[m][n]
                likelihood = 0.
                for c in xrange(self.C):
                    likelihood += rhos[c][u]*thetas[m][c]
                pplex += math.log(likelihood) / self.NU
        return math.exp(0. - pplex)

    def betap(self,x,(a,b)):
        B = 0
        if x == 0.0:
            x = single_step
        if x == 1.0:
            x = 1.0 - single_step
        try:
            B = math.gamma(a+b) / ( math.gamma(a) * math.gamma(b) )
        except Exception as e:
            print x,a,b
            raise e
        return B * (x**(a-1)) * ((1-x)**(b-1))

    def choice(self,p):
        r = random.random()
        s = 0.0
        for i in xrange(len(p)):
            s += p[i]
            if s >= r: return i
            '''
            why it'll be so slow when coded as follow:
            if s >= r: return i
            '''
utlda = UserTimelineLDA(corpus)


























