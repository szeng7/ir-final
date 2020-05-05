import re
from nltk.stem.snowball import SnowballStemmer

infection_words = ['get', 'got', 'recov', 'have', 'had', 'has', 'catch', 'cure', 'infect', 'rest', 'wors', 'weaken', 'weak']
possession_words = ['bat', 'civet', 'pangolin', 'snake', 'epidem', 'sick', 'market', 'pandem', 'posit']
concern_words = ['afraid', 'worri', 'scare', 'fear', 'nervous', 'dread','terrifi']
vaccination_words = ['vaccin', 'drug']
symptom_words = ['fever', 'chill', 'shake', 'muscl', 'headach', 'sore', 'throat', 'tast', 'loss', 'smell', 'cough', 'short', 'breath', 'difficulti']
cdc_words = ['fatal', 'case', 'exposur', 'communiti', 'spread', 'transmiss', 'contact', 'trace', 'drive', 'test', 'droplet', 'epidem', 'flatten', 'isol', 'n95', 'quarantin', 'self', 'distanc', 'ventil']
positive_emoticons = [':)', ':D']
negative_emoticons = [':(', ':/']

stemmer = SnowballStemmer('english')

def count_infection_words(tweet_content):
    count = 0
    for word in tweet_content:
        word = stemmer.stem(word)
        if word in infection_words:
            count += 1
    return count

def count_cdc_words(tweet_content):
    count = 0
    for word in tweet_content:
        word = stemmer.stem(word)
        if word in cdc_words:
            count += 1
    return count

def count_possession_words(tweet_content):
    count = 0
    for word in tweet_content:
        word = stemmer.stem(word)
        if word in possession_words:
            count += 1
    return count

def count_concern_words(tweet_content):
    count = 0
    for word in tweet_content:
        word = stemmer.stem(word)
        if word in concern_words:
            count += 1
    return count

def count_vaccination_words(tweet_content):
    count = 0
    for word in tweet_content:
        word = stemmer.stem(word)
        if word in vaccination_words:
            count += 1
    return count

def count_symptom_words(tweet_content):
    count = 0
    for word in tweet_content:
        word = stemmer.stem(word)
        if word in symptom_words:
            count += 1
    return count

def count_positive_emoticons(tweet_content):
    count = 0
    for word in positive_emoticons:
        if word in tweet_content:
            count += 1
    return count

def count_negative_emoticons(tweet_content):
    count = 0
    for word in negative_emoticons:
        if word in tweet_content:
            count += 1
    return count

def count_mentions(tweet_content):
    return len(re.findall('^@\S+', str(tweet_content)))

def count_hashtags(tweet_content):
    return len(re.findall('^#\S+', str(tweet_content)))

def contains_url(tweet_content):
    return bool(re.search('http[s]?: // (?:[a-zA-Z] |[0-9] |[$-_ @.& +] |[! * \(\),] | (?: %[0-9a-fA-F][0-9a-fA-F]))+', str(tweet_content)))

def determine_length(tweet_content):
    return len(tweet_content)