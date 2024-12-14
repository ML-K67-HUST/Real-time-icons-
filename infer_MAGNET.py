import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import reuters, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import WordNetLemmatizer
import re
import nltk
from MAGNET_2 import MAGNET, load_checkpoint

# Preprocessing functions
stop_words = set(stopwords.words('english'))
stop_words.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across', 'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\W", re.I)

stemmer = SnowballStemmer("english")

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    lemSentence = ""
    for word in sentence.split():
        lem = lemmatizer.lemmatize(word)
        lemSentence += lem
        lemSentence += " "
    lemSentence = lemSentence.strip()
    return lemSentence

def preprocess_text(text):
    text = text.lower()
    text = cleanHtml(text)
    text = cleanPunc(text)
    text = keepAlpha(text)
    text = removeStopWords(text)
    text = lemmatize(text)
    return text

# Load and preprocess data
docs = reuters.fileids()
labels = [reuters.categories(doc) for doc in docs]
texts = [reuters.raw(doc) for doc in docs]
mlb = MultiLabelBinarizer()
bin_labels = mlb.fit_transform(labels)
label_names = mlb.classes_

df = pd.DataFrame({'text': texts})
for i, label in enumerate(label_names):
    df[label] = bin_labels[:, i]

# Store raw texts for display
raw_texts = df['text'].values

# Preprocess texts
df['text'] = df['text'].apply(preprocess_text)

X = df['text'].values
y = df.iloc[:, 1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

# Tokenizer
tokenizer = Tokenizer(num_words=20000, oov_token='<UNK>')
tokenizer.fit_on_texts(X_train)
sequences_text_train = tokenizer.texts_to_sequences(X_train)
sequences_text_test = tokenizer.texts_to_sequences(X_test)

X_train = torch.from_numpy(pad_sequences(sequences_text_train, maxlen=70))
X_test = torch.from_numpy(pad_sequences(sequences_text_test, maxlen=70))

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Load glove embedding
glove_embeddings = {}
with open('glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        glove_embeddings[word] = vector

VOCAB_SIZE = len(tokenizer.index_word) + 1
glove_embedding_matrix = torch.zeros(VOCAB_SIZE, 300)

unk = 0
for i in range(1, VOCAB_SIZE):
    word = tokenizer.index_word[i]
    if word in glove_embeddings.keys():
        glove_embedding_matrix[i] = torch.from_numpy(glove_embeddings[word]).float()
    else:
        unk += 1

# Create adjacency matrix
def create_adjacency_matrix_cooccurance(data_label):
    cooccur_matrix = np.zeros((data_label.shape[1], data_label.shape[1]), dtype=float)
    for y in data_label:
        y = list(y)
        for i in range(len(y)):
            for j in range(len(y)):
                if y[i] == 1 and y[j] == 1:
                    cooccur_matrix[i, j] += 1
    row_sums = data_label.sum(axis=0)

    for i in range(cooccur_matrix.shape[0]):
        for j in range(cooccur_matrix.shape[0]):
            if row_sums[i] != 0:
                cooccur_matrix[i][j] = cooccur_matrix[i, j] / row_sums[i]
            else:
                cooccur_matrix[i][j] = cooccur_matrix[i, j]

    return cooccur_matrix

adj_matrix = create_adjacency_matrix_cooccurance(y_train.numpy())
adj_matrix = torch.tensor(adj_matrix)

# Create label embeddings
glove_label_embedding = torch.zeros(len(label_names), 300)
for index, label in enumerate(label_names):
    wrds = label.split('-')
    for l in wrds:
        if l in glove_embeddings.keys():
            glove_label_embedding[index] += torch.from_numpy(glove_embeddings[l])
    glove_label_embedding[index] = glove_label_embedding[index] / len(wrds)

def infer(model, sample_text, glove_label_embedding):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    glove_label_embedding = glove_label_embedding.to(device)

    # Preprocess the sample text
    inputs = sample_text.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inputs, glove_label_embedding)
        predictions = torch.sigmoid(outputs).round().cpu().numpy()

    predicted_labels = [label_names[i] for i, pred in enumerate(predictions[0]) if pred == 1]
    return predicted_labels

# Load the best model
ckpt_path = 'MAGNET_best_model_final.pt'
model = MAGNET(300, 250, adj_matrix, glove_embedding_matrix, heads=8)
model = load_checkpoint(model, ckpt_path)

# Take a sample from the test set
sample_index = np.random.randint(0, len(X_test))
sample_text = X_test[sample_index]
original_text = raw_texts[sample_index]  # Get the original raw text
true_labels = [label_names[i] for i, val in enumerate(y_test[sample_index]) if val == 1]

# Run inference
predicted_labels = infer(model, sample_text, glove_label_embedding)

print(f"Sample Text: {original_text}")
print(f"True Labels: {true_labels}")
print(f"Predicted Labels: {predicted_labels}")