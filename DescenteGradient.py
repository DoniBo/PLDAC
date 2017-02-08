#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:04:15 2017

@author: Ines
"""
import numpy as np
import numpy.random as rand
from sklearn.datasets import load_svmlight_file

class PerceptronGradient:
    def __init__(self,dimension,eps): #eps = pas
        self.dimension=dimension
        self.eps = eps
        self.teta = rand.random(self.dimension)
            
        
    #Permet de calculer la prediction sur x
    def predict(self,x):
        print(self.teta)
        print(x.toarray().transpose())
        res = int(x.dot(self.teta))
        print(res)
        return abs(res)
        
    #Mise a jour de teta
    def train(self,data):
        for i in range(100):
            j=rand.randint(0,data.shape[0]-1)
            self.teta = self.teta - self.eps*self.gradient(data,j) #*self.delta(data)
            
    #Calcul de l'erreur (Fonction Objective)
    def delta(self, data): 
        error = 0
        for i in range(data.shape[0]):
            x = data.getrow(i)
            pred = self.predict(x) #prediction
            y = data.getY(i) #etiquette reelle
            error += (pred-y)**2
            
        return error
    
    #calcul du gradient
    def gradient(self, data, i):
        x = data.getrow(i)
        y = data[i]
        res = x * (self.predict(x)-y)
                
        return 2*res
        


Xs,Ys=load_svmlight_file("vowel.scale.all")

dim=Xs.shape[1]
p=PerceptronGradient(dim,0.01)
p.train(Xs)


