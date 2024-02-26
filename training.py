import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

with open("intents.json", "r") as f:
    intents = json.load(f)
    
words = []
tags = []
pattern_tag = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    
    for pattern in intent["patterns"]:
        tokenized_pattern = tokenize(pattern)
        words.extend(tokenized_pattern)
        pattern_tag.append((tokenized_pattern, tag))
        
ignore_words = ["?", "!", ".", ","]
stemmed_words = [stem(word) for word in words if word not in ignore_words]
sorted_stemmed_words = sorted(set(stemmed_words))

x_train = []
y_train = []

for (tokenized_pattern, tag) in pattern_tag:
    bag = bag_of_words(tokenized_pattern, sorted_stemmed_words)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)