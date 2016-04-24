import os
import tweepy

CONSUMER_KEY = os.environ['CONSUMER_KEY']
CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
ACCESS_KEY = os.environ['ACCESS_KEY']
ACCESS_SECRET = os.environ['ACCESS_SECRET']
MESSAGE_TEMPLATE = 'Good Morning! Right now there are {} Potentially Hazardous Asteroids over us. Enjoy your day!'

def status_update(status, n_pha, media):
    """Updates status, with a potential orbit plot"""
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth)

    message = MESSAGE_TEMPLATE.format(n_pha)
    if media:
        api.update_with_media(media, status=message)
    else:
        api.update_status(message)
    return "Tweet Published"
