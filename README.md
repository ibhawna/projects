# FakeNews RNN

Fake news is a form of indoctrination that disseminates false information through printed media, televised news outlets, and online social networks. Lies can spread very quickly in a world that is increasingly more interconnected. As social media platforms have grown, fake news has become more prevalent online and has been used for a variety of political and economic goals. The credibility of serious media coverage is damaged by fake news, and it is more challenging for journalists to cover important media articles. Citizens can very easily become infected with fake news using misleading language, and they will spread it without carrying out any fact-checking. The amount of data that is constantly being added to the Internet has expanded to unfathomable heights throughout time, enabling it to serve as a platform for a great deal of undesired, false, and misleading material that may be created by anyone. The process of detecting fake news is drawn out, tiresome, and time-consuming, yet there are numerous conversations and platforms that offer us some assistance. 

Numerous algorithms have been developed as a result to autodetect publications and sites that are false. It appears out that false information can be identified using a dataset of news stories that have been rated as either credible or not. This project uses a recurrent neural network (RNN) to detect potential fake news in articles. We have used the data from the Kaggle. Numerous news stories with their ids, titles, and authors make up this data. Words are first passed on to an embedding layer. The words are first passed to an embedding layer, which is necessary since the input data requires a more effective model beyond one encoded vectors due to the numerous tens of thousands of words in the data. The new hidden layers are provided to LSTM cells once input words are sent to old embedding layer. The LSTM cells expand the network's recurrent connections and enable us to incorporate details about the word order in the data. A sigmoid output layer receives the LSTM outputs. With the help of the sigmoid function and sentiment values ranging from 0 to 1, the result is predicted. Additionally, various model parameters have been adjusted and documented for the best outcomes to guarantee the accuracy of the prediction. 

## RNN Architecture
This project applies a recurrent neural network (RNN) together with long short term memory (LSTM) to identify fake news. A recurrent neural network is an artificial neural network architecture where connections between nodes form a directed graph along a sequence. RNNs can use their internal state (memory) to process sequences of inputs, and therefore, exhibit temporal dynamic behavior for a time sequence. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. LSTMs are an improvement of the RNNs, and are quite useful when our neural network needs to switch between remembering recent things, and things from a long time ago.

<img src="https://raw.githubusercontent.com/minhkhang1795/FakeNews_RNN/master/assets/network_diagram.png" width=50%>
The recurrent neural network (RNN) together with long short term memory (LSTM).

Image Credit: [Intro to Deep Learning with PyTorch by Udacity](https://in.udacity.com/course/deep-learning-pytorch--ud188).

The layers are as follows:

1. An embedding layer that converts our word tokens (integers) into embeddings of a specific size.
2. An LSTM layer defined by a hidden_state size and number of layers
3. A fully-connected output layer that maps the LSTM layer outputs to a desired output_size
4. A sigmoid activation layer which turns all outputs into a value 0-1; return only the last sigmoid output as the output of this network.

The data to train the neural network were obtained from the [Fake News Competition on Kaggle](https://www.kaggle.com/c/fake-news). The dataset which contains 20,800 articles with 267,449 unique words is split into training (80%), validation (20%) and test (20%) sets. As there are approximately 267,449 words in our vocabulary, a word embedding layer is used as a lookup table. In the embedding layer, words are represented by dense vectors where each vector represents the projection of the word into a continuous vector space. The position of a word within the vector space is learned from the text and is based on other surrounding words. The position of a word in the learned vector space is referred to as its embedding.

## Instructions
To train the model, clone this project and open the Jupyter Notebook `FakeNews_RNN.ipynb`
```
git clone https://github.com/minhkhang1795/FakeNews_RNN.git
```

The data can be obtained from the Fake News Competition on Kaggle, which can be downloaded [here](https://www.kaggle.com/c/8317/download-all). Next, extract the downloaded file to the `FakeNews_RNN` folder to obtain two files: 
- **train.csv**: A full training dataset with the following attributes:
  - id: unique id for a news article
  - title: the title of a news article
  - author: author of the news article
  - text: the text of the article; could be incomplete
  - label: a label that marks the article as potentially unreliable
    - 1: unreliable
    - 0: reliable
  
- **test.csv**: A testing training dataset with all the same attributes at **train.csv** without the label.

The test set is for Kaggle submission. Therefore, we need to create our own test set by splitting the training data into training (80%), validation (20%) and test (20%) sets, which is instructed in the Jupyter Notebook.

Some hyperparameters of the model we could change:
```python
embedding_dim = 200
hidden_dim = 256
n_layers = 3
```
These hyperparameters together define our model (read more about the RNN Architecture in the section above).

