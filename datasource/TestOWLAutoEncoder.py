# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:13:16 2018

@author: Giuseppe
"""

from OntologyLoader import *
from Autoencoder import *
kb= KnowledgeBase('file://C:/Users/Giuseppe/Documents/ontos/financialxml.owl')
list=kb.computeProjections()
print (len(list[1]))
#print(kb.computeEntropy())

encoder= Autoencoder(217)
encoder.fit(list, 100)
