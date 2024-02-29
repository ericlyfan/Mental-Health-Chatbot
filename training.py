import json
import numpy as np
import torch
import torch.nn as nn
from nltk_utils import tokenize, stem, bag_of_words
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

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
        
ignore_words = ["?", "!", ".", ",", ";", ":"]
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

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


batch_size = 16
hidden_size = 16
learning_rate = 0.001
num_epochs = 1000
output_size = len(tags)
input_size = len(sorted_stemmed_words)


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # backwards and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")
        
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": sorted_stemmed_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training complete. File saved to {FILE}")