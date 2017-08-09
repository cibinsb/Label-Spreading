import numpy as np
from  scipy import sparse

class LabelSpreading(object):

    def __init__(self,similarity_matrix):
        self.alpha=0.99
        self.W=similarity_matrix
        sum=self.W.sum(axis=0)
        self.D = np.sqrt(sparse.diags((1.0 / sum), offsets=0))
        self.S = self.D.dot(self.W).dot(self.D)

    def train(self,x,y):
        samples = self.W.shape[0]
        classes = y.max() + 1
        self.F=np.zeros((samples,classes))
        self.Y=np.zeros((samples,classes))
        self.Y[x,y]=1
        for i in xrange(0,30):
            self.F=self.alpha*self.S.dot(self.F)+(1-self.alpha)*self.Y
        return self

    def predict(self,x):
        #random walk and node assignment 
        #applying some  regularization
        d=np.sum(self.F[x], axis=1)
        return np.argmax((self.F[x].T/d).T, axis=1)