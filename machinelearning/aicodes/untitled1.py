#Step1:- conda install tweepy
#Step2:- conda install -c conda-forge textblob

from textblob import TextBlob
hlo=TextBlob("Vipul used to be very happy until someone unnecessary pokes him")
#hlo="Vipul used to be very happy until someone unnecessary pokes him"

print(hlo.tags)
print(hlo.words)
print(hlo.sentiment.polarity)

import tweepy
from textblob import TextBlob

consumer_key='Z9lbmPWP4sLOSeXWAEWnE3ipv'
consumer_secret='wBN2vlDlJ2pz2qi9M4gvbFcRDyQjnxx0AVPU4QaGLwH6WbUAlX'

access_token='162988906-MHvzgWxcEFY6WSO9eCd3qe6EPFTy3CL11nZJiMWs'
access_token_secret='sXQiJPqsZEFq2QXuBfJC2WTwHoQxynhznp0gmThtg2CvQ'

auth=tweepy.OAuthHandler(consumer_key,
                         consumer_secret)
auth.set_access_token(access_token, 
                      access_token_secret)
api=tweepy.API(auth)

public_tweets=api.search('Satue of Unity')

for tweet in public_tweets:
    print(tweet.text)
    analysis=TextBlob(tweet.text)
    print(analysis.sentiment)






