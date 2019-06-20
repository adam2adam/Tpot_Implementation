# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:42:19 2019

@author: A321545
"""

# Import scikit-learn dataset library
import pandas as pd
# import Data_preparation  # importing Data_preperation module
# Import train_test_split function
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load dataset
# Digits is a dataset of handwritten digits.
# Each feature is the intensity of one pixel of an 8 x 8 image.
digits = load_digits()

# View the first observation's feature values as a matrix
##digits.images[0]
# Visualize the first observation's feature values as an image
##plt.gray() 
##plt.matshow(digits.images[0]) 
##plt.show()


# Split dataset into training set and test set
# 70% training and 25% test
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

# ////////////////
# TPOT
tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# It generates new code in the file 'tpot_data_pipeline.py'
# If we ran TPOT for more generations, then the score should improve further.
tpot.export('tpot_data_pipeline.py')
# ////////////////

