# Mental Health Chatbot

## Project Function & Interest

Welcome to Baymax, your personal mental health chatbot! I'm sure that we've all felt a little emotionally down sometimes or maybe we just feel curious/confused in general about our feelings and mental health. This was my main motivation for creating Baymax, my retrieval based python chatbot. While Baymax is not perfect, as he is designed on a closed-domain architecture, and is unable to generate new context-specific responses for the user's input directly, Baymax can still offer a quick and accessible type of assistance and support to it's users.

Baymax is trained on an 'Mental Health JSON dataset' and is capable of providing basic assistance and responses to user concerns, questions, and input. For example, you can say things like: "My job is stressing me out" or "I've been feeling really down lately." and Baymax is able to retrieve a suitable response to your input.

### About & Techniques

- Natural Language Processing (NLP): I utlized NLP techniques including tokenization, stemming and the creation of a bag-of-words (BoW) language model in order to convert the 'Mental Health JSON dataset' into a format that the chatbot is able to use.

- Neural Network (Feed Forward) Architecture: This neural network (NN) contains one input layer, two hidden layers, and one output layer in total. The input layer takes the user's input, which has already been preprocessed (tokenization, stemming, conversion to BoW format) and feeds it into the NN. The two hidden layers are responsible for teaching the NN patterns and relationships from the dataset by using the ReLU (Rectified Linear Unit) activation function. The output layer corresponds to the number of intents the NN is trained with in the previous step. The softmax function is applied to the output layer to obtain the closest (most probable) intent, and a response is retrieved based on that predicted intent.

### Dependencies

- PyTorch
- JSON
- NumPy
- Natural Language Toolkit (NLTK)

### Reflection

While developing Baymax, I am aware that creating a chatbot that is meant to engage with individuals on such a sensitive and serious topic such as mental health comes with serious responsibilites/consequences, Therefore, it is very important to acknowledge that the responses Baymax may generate is not comprehensive, nor is it always accurate. Baymax was created by myself only out of passion and is not based on any credible medical research or data. Please make sure to seek professional help and support in the event of a real-life mental health issue.

In the future, I hope to update and improve Baymax by implementating a more user friendly interface such as a GUI, website, or mobile application to improve and ease user experience. Additionally, I want to pursue the possibilites of converting Baymax into a hybrid model with both retrieval and generative capabilities in order to further improve Baymax's ability to generate new context-specific responses in real-time.
