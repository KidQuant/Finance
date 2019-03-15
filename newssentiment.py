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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

#create a single string for each date (since we only want to look at word counts)
news_combined = ''
for row in range(0,len(df.index)):
    news_combined +=' '.join(str(x).lower().strip() for x in df.iloc[row,2:27])

vectorizer = CountVectorizer()
news_vect = vectorizer.build_tokenizer()(news_combined)
word_counts = pd.DataFrame([[x,news_vect.count(x)] for x in set(news_vect)], columns = ['Word', 'Count'])

from wordcloud import WordCloud
wordcloud = WordCloud().generate(news_combined)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

#Lower max_font_size
word_cloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


