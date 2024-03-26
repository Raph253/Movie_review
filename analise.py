import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import apply_features

nltk.download('movie_reviews')

# Coleta de dados
positive_reviews = [(words, 'positive') for words in movie_reviews.words(categories='pos')]
negative_reviews = [(words, 'negative') for words in movie_reviews.words(categories='neg')]

# Preparação do conjunto de treinamento
train_set = positive_reviews[:800] + negative_reviews[:800]

# Função para extrair features
def extract_features(words):
    return {word: True for word in words}

def classificacao(classification):
    if classification == "positive":
        return "positiva"
    elif classification == "negative":
        return "negativa"
    else:
        return "neutra"

# Aplicação das features
train_features = apply_features(extract_features, train_set)

# Treinamento do classificador
classifier = NaiveBayesClassifier.train(train_features)

# Exemplo de classificação
new_sentence = str(input("Faca sua analise: "))
new_features = extract_features(new_sentence.split())
classification = classifier.classify(new_features)
print(f"Classificacao: {classificacao(classification)}")
