#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:15:12 2017

@author: paulbonnier
"""

import torch
from sklearn.datasets import load_svmlight_file
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import random

############################
### PARTIE FOURNIE DEBUT ###
############################

def reindex_categories(Ys):
    index={}
    for i in Ys:
        if (not i in index):
            index[i]=len(index)
    if (len(index)==max(Ys)+1):
        return Ys

    logging.info("Categories are not contiguous: reindexing categories...")
    nys=[]
    for i in Ys:
        nys.append(index[i])
    return nys

Xs,Ys=load_svmlight_file("vowel.scale.all")
Ys=reindex_categories(Ys)
Xs=Xs.todense()
input_size=Xs.shape[1]
Xs=torch.Tensor(np.array(Xs))
Ys=torch.LongTensor(np.array(Ys,dtype=np.int))

##########################
### PARTIE FOURNIE FIN ###
##########################

# Définir le modèle: Linéarisation des données entrantes
# de taille 10 ==> sortie taille 11
model=nn.Linear(10,11)

# On crée le gradient associé au modèle (pas(step) par défaut)
optimizer=optim.Adam(model.parameters())


t=0
# Pendant longtemps
while(t<10):
    # Init du gradient
    optimizer.zero_grad()
    
    # On entraîne notre modèle sur Xs[i] tié aléatoirement
    i = random.randint(0,len(Xs)-1)
    input = Variable(Xs[i].unsqueeze(0),requires_grad=True)
    
    # On récupère la variable réelle qu'on souhaite obtenir
    vreelle= Variable(torch.LongTensor([Ys[i]]))
    
    # Fprédiction
    output=model(input)
    
    # Loss
    delta=nn.NLLLoss()
    erreur=delta(output,vreelle)
    
    # On "stock" le loss qu'on veut minimiser
    erreur.backward()
    
    # On change le pas du gradient
    optimizer.step()
    t += 1
print(erreur)














