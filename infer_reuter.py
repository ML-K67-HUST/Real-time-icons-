import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from model.ANN import ANN
from model.LSTM import LSTMClassifier
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import reuters, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import WordNetLemmatizer
import re
import nltk

nltk.download('stopwords')
from model.MAGNET import MAGNET, load_checkpoint
def infer_dnn_and_lstm(test_text):
    ## DNN and LSTM
    model_path1 = './checkpoint/model_checkpoint.pth'
    model_path2 = './checkpoint/lstm_reuters_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = ANN(n_in=16909, n_out=90).to(device)
    model2 = LSTMClassifier(
            input_size=16909,
            hidden_size=256,
            num_layers=2,
            n_out=90
        ).to(device)

    checkpoint1 = torch.load(model_path1, map_location=device)
    if 'model_state_dict' in checkpoint1:
        model1.load_state_dict(checkpoint1['model_state_dict'])
    else:
        model1.load_state_dict(checkpoint1)
    checkpoint2 = torch.load(model_path2, map_location=device)
    if 'model_state_dict' in checkpoint2:
        model2.load_state_dict(checkpoint2['model_state_dict'])
    else:
        model2.load_state_dict(checkpoint2)

    model1.eval()
    model2.eval()

    tfidf_path = './checkpoint/tfidf_vectorizer.pkl'
    mlb_path = './checkpoint/multilable.pkl'
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)

    # Test text
    # test_text = [
    #     "JOHOR BAHRU: By mid-2025, Singaporeans and other foreign travellers may be able to clear Johor land checkpoints with just QR codes without needing to show their passports. A senior official of the Johor state government Lee Ting Han said on Wednesday (Dec 11) that Malaysian authorities are aiming for QR code immigration clearance to be expanded to Singaporeans and other foreign passport holders by the middle of next year."
    # ]

    # text = input('Input your text: ')
    # test_text = []
    # test_text.append(text)
    test_text = ['My honey, you are so hot !']

    test_text_preprocessed = [text.lower() for text in test_text]
    X_test = tfidf.transform(test_text_preprocessed)
    X_test_tensor = torch.FloatTensor(X_test.toarray()).to(device)

    with torch.no_grad():
        outputs1 = model1(X_test_tensor)
        outputs2 = model2(X_test_tensor)
    probabilities1 = outputs1
    probabilities2 = outputs2
    top5_values_DNN, top5_indices_DNN = torch.topk(probabilities1, k=5)
    top5_values_LSTM, top5_indices_LSTM = torch.topk(probabilities2, k=5)


    top5_values_DNN = top5_values_DNN.cpu()
    top5_indices_DNN = top5_indices_DNN.cpu()
    top5_values_LSTM = top5_values_LSTM.cpu()
    top5_indices_LSTM = top5_indices_LSTM.cpu()
    # print("\nTop 5 Predicted Labels and Values:")
    result = {'dnn':[],'lstm':[]}
    for i in range(len(test_text)):
        # print('--DNN--')
        for label_idx, prob in zip(top5_indices_DNN[i], top5_values_DNN[i]):
            label = mlb.classes_[label_idx]
            # print(f"  {label}: {prob:.4f}")
            result['dnn'].append(
                {
                    'label':label,
                    'prob':round(prob.item(),4)
                }
            )
        # print('--LSTM--')
        for label_idx, prob in zip(top5_indices_LSTM[i], top5_values_LSTM[i]):
            label = mlb.classes_[label_idx]
            result['lstm'].append(
                {
                    'label':label,
                    'prob':round(prob.item(),4)
                }
            )
            # print(f"  {label}: {prob:.4f}")
    return result

    
## MAGNET
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
# print(f'x test here: {X_test}')
# print(f'x test type: {type(X_test)}')
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
with open('./glove/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
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
        predictions = outputs.cpu().numpy()[0]

    # Sort predictions and get the top 5 labels
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_labels = [label_names[i] for i in top_indices]
    top_values = predictions[top_indices]

    return top_labels, top_values

def infer_raw(input):
    ckpt_path = './checkpoint/MAGNET_best_model_final.pt'
    model = MAGNET(300, 250, adj_matrix, glove_embedding_matrix, heads=8)
    model = load_checkpoint(model, ckpt_path)
    input =np.array([input])
    sequence = tokenizer.texts_to_sequences(input)
    seq_tensor = torch.from_numpy(pad_sequences(sequence, maxlen=70))
    top_labels, top_values = infer(model, seq_tensor[0], glove_label_embedding)
    # print(f"Seq tensor: {seq_tensor}")
    # print(f"Top 5 Predicted Labels: {top_labels}")
    # print(f"Top 5 Predicted Values: {top_values}")
    # print('--MAGNET--')
    result = {'magnet':[]}

    for label, values in zip(top_labels, top_values):
        # print(f' {label}: {values:.4f}')
        result['magnet'].append(
            {
                'label':label,
                'value':values
            }
        )
    return result

# print("\nTop 5 Predicted Labels and Values:")
# for i in range(len(test_text)):
#     print('--DNN--')
#     for label_idx, prob in zip(top5_indices_DNN[i], top5_values_DNN[i]):
#         label = mlb.classes_[label_idx]
#         print(f"  {label}: {prob:.4f}")
#     print('--LSTM--')
#     for label_idx, prob in zip(top5_indices_LSTM[i], top5_values_LSTM[i]):
#         label = mlb.classes_[label_idx]
#         print(f"  {label}: {prob:.4f}")

def infer_reuter(test_text):
    mag = (infer_raw(test_text[0]))
    dnn_lstm =(infer_dnn_and_lstm(test_text))
    mag.update(dnn_lstm)
    return mag
# print(infer_reuter('You are so hot'))