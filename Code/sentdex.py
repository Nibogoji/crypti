from newsapi.newsapi_client import NewsApiClient
import csv
import pandas as pd
from textblob import TextBlob
from dateutil import parser
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

api = NewsApiClient(api_key = '5159fdefbd7c45b29d4120186c7b35b6')

all_articles = api.get_everything( q = 'bitcoin',
                                   sources = 'bbc-news,metro,crypto-coins-news, financial-times,twitter,reddit, '
                                             'business-insider,reuters, bloomberg, cnbc, fortune, cbc-news',
                                   from_param= '2020-12-26',
                                   language = 'en')


news = pd.DataFrame(all_articles['articles'])
news['polarity'] = news.apply(lambda x: TextBlob(x['description']).sentiment.polarity, axis=1)
news['subjectivity'] = news.apply(lambda x: TextBlob(x['description']).sentiment.subjectivity, axis=1)
news['date'] = news.apply(lambda x: parser.parse(x['publishedAt']).strftime('%Y-%m-%d'), axis=1)


news_mean = news.groupby(by = 'date').mean()
news_mean.index = pd.to_datetime(news_mean.index)

btc_price = pd.read_csv('C:\\Users\\stesc\\Desktop\\crypti\\Data\\BTCUSDT.csv', index_col=0)
btc_price.index = pd.to_datetime(btc_price.index)
btc_price = btc_price.resample('12H').mean()
a = pd.DataFrame(btc_price.loc['2020-12-25 00:00:00':'2021-01-22 00:00:00']['close'])
sentdex_price = a.join(news_mean)
sentdex_price.fillna(0,inplace=True)

fig, axs = plt.subplots(2,sharex=True)
axs[0].plot(sentdex_price['close'], label = 'Close')
axs[1].plot(sentdex_price['polarity'], label = 'polarity')
axs[1].plot(sentdex_price['subjectivity'],label = 'subjectivity')
plt.legend()
plt.gcf().autofmt_xdate()
fig.show()

############################################   TWITTER   ############################################

import codecs
from bs4 import BeautifulSoup
import requests
import tweepy
from textblob import TextBlob
import sys
import csv
from fake_useragent import UserAgent


consumer_key= 'DVP3CrFNawbFAYAxp3jshAxXw'
consumer_secret= 'MjV1J4gS4GnrZw58iCBwfLmJkGaLFHMA1mDhW4gzZKEe1i3Bm2'
access_token='1353699102425554944-E1W3nnS5oGEdl3w7XZvs5tiUANftVb'
access_token_secret= 'EvE9re7W7g7JciJT8SGkxsaion4WJw1faefEuVJtvPHdu'
#tweepy library to authenticate our API keys
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


url = 'https://cryptrader.com/dashboard/main/redddit'
#Picking random useragent / telling user and setting below to use
print("Grabing a random useragent from random useragent API....")
# get a random user agent
randuserAgent = UserAgent(verify_ssl=False).random
headers = {'User-Agent': randuserAgent} # spoof user agent to stop the block
page = requests.get(url, headers=headers) # grab the page / html
print(page.content)
print(page.status_code)
soup = BeautifulSoup(page.content, 'html.parser')
tabulka = soup.find("table", {"class" : "data-table"}) #grabbing the table from our 'api'
records = [] # store all of the records in this list

for row in tabulka.findAll('tr'):
    col = row.findAll('td')
    name = col[0].string.strip()
    symbol = col[1].string.strip()
    tweetsLastHour = col[2].string.strip()
    try:
        change = col[3].string.strip()
    except:
        change = "NULL"
#Search for tweets that include the symbol+name using tweepy
    public_tweets = api.search("#" + name)


#############################   CRYPTO PANIC ###############################

import requests, json
import pprint

r = requests.get('https://cryptopanic.com/api/v1/posts/?auth_token=1ee777e8ba2df0b94b95be01e2e1208d9257efaa&currencies=BTC')
# pprint.pprint(r.json())
#

dates = []
authors = []
titles = []
eth_news = {}
for i in range(len(r.json()['results'])):

    dates.append(r.json()['results'][i]['created_at'])
    authors.append(r.json()['results'][i]['domain'])
    titles.append(r.json()['results'][i]['title'])
    eth_news[i] = {}

news_info = [i for i in zip(dates,authors,titles)]


for n in range(len(news_info)):

    eth_news[n]['date'] = news_info[n][0]
    eth_news[n]['from'] = news_info[n][1]
    eth_news[n]['title'] = news_info[n][2]


