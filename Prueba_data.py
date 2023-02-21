# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:41:58 2023

@author: Julian Garcia
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # viz
import matplotlib.pyplot as plt # viz
from scipy import stats
import json
from typing import List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics, linear_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')



#train_df = pd.read_csv('/kaggle/input/beth-dataset/labelled_training_data.csv')
#test_df = pd.read_csv('/kaggle/input/beth-dataset/labelled_testing_data.csv')
#validation_df = pd.read_csv('/kaggle/input/beth-dataset/labelled_validation_data.csv')

train_df = pd.read_csv('labelled_training_data.csv')
test_df = pd.read_csv('labelled_testing_data.csv')
validation_df = pd.read_csv('labelled_validation_data.csv')


assert train_df.columns.all() == test_df.columns.all() == validation_df.columns.all()

train_df.dtypes