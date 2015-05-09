# -*- coding: utf-8 -*-

import os
import random
import math
from numpy import random as nprand
import copy as cp
import corpus as cputil

corpus = cputil.load_corpus("corpus_filtered.final")
corpus = filter(lambda c:len(c[0])>48,corpus)


class UserLDA:
    def __init__(self,corpus,alpha=0.1,beta=0.01,gamma=0.01,K=20,n_iter=300):
        self.corpus = corpus
        self.corpus_word = [w for w,_ in self.corpus]
        self.corpus_user = [u for _,u in self.corpus]
        self.M = len(self.corpus)
        self.NW = sum(map(len,self.corpus_word))
        self.NU = sum(map(len,self.corpus_user))
        
        self.wdic = list(set([w for d in self.corpus_word for w in d]))
        self.wsize = len(self.wdic)
        self.udic = list(set([u for d in self.corpus_user for u in d]))
        self.usize = len(self.udic)
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

        self.n_m = [[0 for k in range(self.K)] for m in range(self.M)]
        self.n_m_sum = [0 for m in range(self.M)]
        self.n_z = [{w:0 for w in self.wdic} for k in range(self.K)]
        self.n_z_sum = [0 for k in range(self.K)]

        self.n_t = [[0 for c in range(self.C)] for m in range(self.M)]
        self.n_t_sum = [0 for m in range(self.M)]
        self.n_c = [{u:0 for u in self.udic} for c in range(self.C)]
        self.n_c_sum = [0 for c in range(self.C)]

        for m in range(self.M):
            doc_word = self.corpus_word[m]
            doc_user = self.corpus_user[m]
            for i in range(len(doc_word)):
                w = self.corpus_word[m][i]
                z = self.topics[m][i]
                self.n_m[m][z] += 1
                self.n_z[z][w] += 1
                self.n_m_sum[m] += 1
                self.n_z_sum[z] += 1

            for i in range(len(doc_user)):
                u = self.corpus_user[m][i]
                c = self.communities[m][i]
                self.n_t[m][c] += 1
                self.n_c[c][u] += 1
                self.n_t_sum[m] += 1
                self.n_c_sum[c] += 1

        self.phi = [{w:0 for w in self.wdic} for k in range(self.K)]
        self.rho = [{u:0 for u in self.udic} for c in range(self.C)]
        self.theta = [[0 for k in range(self.K)] for m in range(self.M)]
        self.nstats = 0

        self.perplexities = []


    def run(self,n_iter=300):
        self.n_iter = n_iter
        for n in range(self.n_iter):
            print "iteration ",n
            self.gibbs_routine()
            self.update_params()
            pplex = self.perplexity()
            print "perplexity: ",pplex
            self.perplexities.append(pplex)

    def gibbs_routine(self):
        for m in range(self.M):
            #print "sampling document %s/%s ..." % (m,self.M)
            dw = self.corpus_word[m]
            du = self.corpus_user[m]

            for wi in range(len(dw)):
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
            
            for ui in range(len(du)):
                #print "sampling communities %s/%s , %s/%s" % (m,len(self.corpus),i,len(du))
                c = self.communities[m][ui]
                u = self.corpus_user[m][ui]
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


    def sample_topic(self,m,wi):
        z = self.topics[m][wi]
        w = self.corpus_word[m][wi]
        
        msum = self.n_m_sum[m] + self.n_t_sum[m] + self.Kalpha
#        def phik(k):
#            return (self.n_z[k][w] + self.beta) / (self.n_z_sum[k] + self.wsize * self.beta)
#        def thetam(k):
#            return (self.n_m[m][k] + self.n_t[m][k] + self.alpha) / msum)
#        pz = [phik(k) * thetam(k) for k in range(self.K)]
        
        pz = [( self.n_m[m][k] + self.n_t[m][k] + self.alpha ) * ( self.n_z[k][w] + self.beta ) / ( msum * (self.n_z_sum[k] + self.wbeta ) ) for k in range(self.K)]

        _sum = sum(pz)
        pz = [pi/_sum for pi in pz]

        sample = self.choice(pz)
        return sample
        
    def sample_community(self,t,ui):
        c = self.communities[t][ui]
        u = self.corpus_user[t][ui]
        
        msum = self.n_t_sum[t] + self.n_m_sum[t] + self.C * self.alpha
        
        pc = [( self.n_t[t][c] + self.n_m[t][c] + self.alpha ) * ( self.n_c[c][u] + self.gamma ) / ( msum * ( self.n_c_sum[c] + self.ugamma ) ) for c in range(self.C)]
                
        _sum = sum(pc)
        pc = [pi/_sum for pi in pc]

        sample = self.choice(pc)
        return sample

    def update_params(self):
#        print "updating parameters..."
        self.theta = [[(self.n_m[m][k] + self.n_t[m][k] + self.alpha) / (self.n_m_sum[m] + self.n_t_sum[m] + self.Kalpha) for k in range(self.K)] for m in range(self.M)]
        self.phi = [{w:(self.n_z[k][w] + self.beta) / (self.n_z_sum[k] + self.wbeta) for w in self.wdic} for k in range(self.K)]
        self.rho = [{u:(self.n_c[c][u] + self.gamma) / (self.n_c_sum[c] + self.ugamma) for u in self.udic} for c in range(self.C)]

#        for m in range(self.M):
#            for k in range(self.K):
#                self.theta[m][k] = (self.n_m[m][k] + self.n_t[m][k] + self.alpha) / (self.n_m_sum[m] + self.n_t_sum[m] + self.K * self.alpha) 
#
#        for k in range(self.K):
#            for w in self.wdic:
#                self.phi[k][w] = (self.n_z[k][w] + self.beta) / (self.n_z_sum[k] + self.wsize * self.beta)
#
#        for c in range(self.C):
#            for u in self.udic:
#                self.rho[c][u] = (self.n_c[c][u] + self.gamma) / (self.n_c_sum[c] + self.usize * self.gamma)
        self.nstats += 1

    def show_topics(self,num_topics=20,num_words=20):
        topics = [sorted(phi.items(),key=lambda (w,c):c, reverse=True) for phi in self.phi]
        topics = [topic[:num_words] for topic in topics][:num_topics]
        return topics

    def show_communities(self,num_communities=20,num_words=20):
        communities = [sorted(rho.items(),key=lambda (w,c):c, reverse=True) for rho in self.rho]
        communities = [community[:num_words] for community in communities][:num_communities]
        return communities

    def get_phi(self):
        phis = [sorted(phi.items(),key=lambda (w,c):c, reverse=True) for phi in self.phi]
        phis = [dict(phi) for phi in phis]
        return phis

    def get_theta(self):
#        thetas = [[kc for kc in doc] for doc in self.theta]
        return self.theta
        return thetas

    def get_rho(self):
        rhos = [sorted(rho.items(),key=lambda (w,c):c, reverse=True) for rho in self.rho]
        rhos = [dict(rho) for rho in rhos]
        return rhos

    def perplexity(self):
        phis = self.phi
        thetas = self.theta
        rhos = self.rho

        pplex = 0.
        for m in range(self.M):
            doc = self.corpus_user[m]
            for n in range(len(doc)):
                u = self.corpus_user[m][n]
                likelihood = 0.
                for c in range(self.C):
                    likelihood += rhos[c][u]*thetas[m][c]
                pplex += math.log(likelihood) / self.NU
        return math.exp(0. - pplex)

    def choice(self,p):
        r = random.random()
        s = 0.0
        for i in range(len(p)):
            s += p[i]
            if s >= r: return i
            '''
            why it'll be so slow when coded as follow:
            if s >= r: return i
            '''
lda = UserLDA(corpus)


























