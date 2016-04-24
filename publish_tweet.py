import os
import tweepy

CONSUMER_KEY = os.environ['CONSUMER_KEY']
CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
ACCESS_KEY = os.environ['ACCESS_KEY']
ACCESS_SECRET = os.environ['ACCESS_SECRET']
MESSAGE_TEMPLATE = 'Good Morning! Right now there are {} Potentially Hazardous Asteroids over us. Enjoy your day!'

def status_update(status, n_pha, media):
    """Updates status, with a potential orbit plot"""
    CONSUMER_KEY = '1RfuCAPIhU9Daeih2Ed1g7MqU'
    CONSUMER_SECRET = 'oS482yLCzUOihzbBMiHrqt3I8kWgBdtQI3ca8Jyv5uKmJxyttk'
    ACCESS_KEY = '724152610094211072-pSJOOj8FMqLle3h6HPsaEysMCCTxT1V'
    ACCESS_SECRET = 'LTD0l1CQDy7F5Hfko5ddtnBH2hu98q4UwopgvPE2nsC9r'
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth)

    message = MESSAGE_TEMPLATE.format(n_pha)
    if media:
        api.update_with_media(media, status=status)
    else:
        api.update_status(message)
    return "Tweet Published"
