import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

df = pd.read_csv('Finance\data\Combined_News_DJIA.csv')
dj_df = pd.read_csv('Finance\data\DJIA_table.csv')
reddit_df = pd.read_csv('Finance\data\RedditNews.csv')

df.describe()
df.Date = pd.to_datetime(df.Date)
df.head()
df.index = df.Date

dj_df.describe()
dj_df.Date = pd.to_datetime(dj_df.Date)
dj_df.index = dj_df.Date
dj_df.drop(columns = ['Date'], inplace=True)
dj_df = dj_df.sort_values(by = 'Date', ascending=True)
dj_df.head()

