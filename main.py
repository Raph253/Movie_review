import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet.lower())  # Tokenização e conversão para minúsculas
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

# Exemplo de uso:
tweet = "NLTK is awesome! I love natural language processing."
preprocessed_tokens = preprocess_tweet(tweet)
print(preprocessed_tokens)


positive_words = twitter_samples.tokenized('positive_tweets.json')
negative_words = twitter_samples.tokenized('negative_tweets.json')

def calculate_sentiment_score(tokens):
    positive_score = sum(1 for token in tokens if token in positive_words)
    negative_score = sum(1 for token in tokens if token in negative_words)
    return positive_score - negative_score

# Exemplo de uso:
sentiment_score = calculate_sentiment_score(preprocessed_tokens)
print(f"Sentiment Score: {sentiment_score}")
