import wandb
import nltk
# nltk.download('reuters')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import reuters
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from nltk import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import torch
from torch import optim
from torch import nn

from sklearn.metrics import hamming_loss, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer
import os

from torch.utils.data import Dataset, DataLoader

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class MAGNET(nn.Module):
    def __init__(self, input_size, hidden_size, adjacency, embeddings, heads=4, slope=0.01, dropout=0.5):
        super(MAGNET, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.biLSTM = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.adjacency = nn.Parameter(adjacency)
        self.dropout = nn.Dropout(dropout)
        self.edge_weights = nn.Linear(hidden_size * 2 * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(slope)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.heads = heads
        self.transform_dim1 = nn.Linear(input_size, hidden_size * 2, bias=False)
        self.transform_dim2 = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.transform_dimensions = [self.transform_dim1, self.transform_dim2]

    def forward(self, token, label_embedding):
        # BILSTM part
        features = self.embedding(token)
        out, (h, _) = self.biLSTM(features)
        embedding = torch.cat([h[-2, :, :], h[-1, :, :]], dim=1)
        embedding = self.dropout(embedding)

        # GAT PART
        for td in self.transform_dimensions:  # Two Multiheaded GAT layers
            outputs = []
            for head in range(self.heads):
                label_embed = td(label_embedding)
                n, embed_size = label_embed.shape

                label_embed_combinations = label_embed.unsqueeze(1).expand(-1, n, -1)
                label_embed_combinations = torch.cat([label_embed_combinations, label_embed.unsqueeze(0).expand(n, -1, -1)], dim=2)
                e = self.activation(self.edge_weights(label_embed_combinations).squeeze(2))

                attention_coefficients = self.tanh(torch.mul(e, self.adjacency))

                new_h = torch.matmul(attention_coefficients.to(label_embed.dtype), label_embed)
                outputs.append(new_h)
            outputs = self.activation(torch.mean(torch.stack(outputs, dim=0), dim=0))

            label_embedding = outputs
        attention_features = self.dropout(label_embedding)
        attention_features = attention_features.transpose(0, 1)
        predicted_labels = torch.matmul(embedding, attention_features)
        return predicted_labels

def save_checkpoint(model, optimizer, epoch, loss, ckpt_path):
    """
    Save the best model checkpoint
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, ckpt_path)
    print(f"Saved best model to {ckpt_path}")

def load_checkpoint(model, ckpt_path):
    """
    Load the best model checkpoint
    """
    if not os.path.exists(ckpt_path):
        print("No checkpoint found")
        return model

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.5f}")
    return model

def train(model, X_train, label_embedding, y_train,
          total_epoch=200, batch_size=250, learning_rate=0.001,
          ckpt_path='MAGNET_best_model_final.pt'):

    wandb.init(
        project="magnet-classification",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": total_epoch,
            "architecture": "MAGNET"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    label_embedding = label_embedding.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_data = DataLoader(dataset(X_train, y_train), batch_size=batch_size)

    best_loss = float('inf')

    for epoch in range(1, total_epoch + 1):
        running_loss = 0
        y_pred = []
        model.train()

        for index, (X, y) in enumerate(train_data):
            optimizer.zero_grad()
            out = model(X.to(device), label_embedding)
            loss = criterion(out, y.to(device).float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            y_pred.append(torch.sigmoid(out.detach()).round().cpu())
            running_loss += loss.item()

        y_pred = torch.vstack(y_pred)
        f1score = f1_score(y_train, y_pred, average='micro')
        hammingloss = hamming_loss(y_train, y_pred)

        # Save best model
        if running_loss < best_loss:
            best_loss = running_loss
            save_checkpoint(model, optimizer, epoch, running_loss, ckpt_path)

        wandb.log({
            "epoch": epoch,
            "loss": running_loss,
            "hamming_loss": hammingloss,
            "micro_f1_score": f1score
        })

        print(f'epoch:{epoch} loss:{running_loss:.5f} hamming_loss:{hammingloss:.5f} micro_f1score:{f1score:.5f}')

    wandb.finish()

def evaluate(model, X_test, label_embedding, y_test, batch_size=250):
    model.eval()
    test_data = DataLoader(dataset(X_test, y_test), batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    label_embedding = label_embedding.to(device)
    y_pred = []
    test_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for X, y in test_data:
            X = X.to(device)
            y = y.to(device)
            out = model(X, label_embedding)
            loss = criterion(out, y.float())
            test_loss += loss.item()
            y_pred.append(torch.sigmoid(out).round().cpu())

    y_pred = torch.vstack(y_pred)

    # Calculate metrics
    f1_micro = f1_score(y_test, y_pred, average='micro')
    hammingloss = hamming_loss(y_test, y_pred)

    print(f"Test Results:")
    print(f"Loss: {test_loss:.5f}")
    print(f"Micro F1: {f1_micro:.5f}")
    print(f"Hamming Loss: {hammingloss:.5f}")

    return y_pred, test_loss

# Additional code for retraining the model
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from nltk.corpus import reuters, stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk import WordNetLemmatizer
    import re
    import nltk

    # Download NLTK data
    nltk.download('reuters')
    nltk.download('stopwords')
    nltk.download('wordnet')

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
    

    class dataset(Dataset):
        def __init__(self, x, y):
            self.x  = x
            self.y = y
        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

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
    with open('glove/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
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

    # Initialize and train the model
    model = MAGNET(300, 250, adj_matrix, glove_embedding_matrix, heads=8)
    train(model, X_train, glove_label_embedding, y_train, total_epoch=200, ckpt_path='MAGNET_best_model_final.pt')
