import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

#Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, FloatRangeSlider

#spacy for language processing
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

#sklearn for feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#gensim for topic models
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

#topic modle viz
import pyLDAvis
from pyLDAvis.gensim import prepare

#evaluate parameter setting
import statsmodels.api as sm
