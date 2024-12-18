import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from model.ANN import ANN
from model.LSTM import LSTMClassifier

df = pd.read_csv('./preprocessed_data/preprocessed_goemotion.csv')
x = df['Text']
label_columns = df.columns[2:]
y = df[label_columns]

model_path1 = './checkpoint/go_emotion1.pth'
model_path2 = './checkpoint/go_emotion_LSTM.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = ANN(n_in=23523, n_out=28).to(device)
model2 = LSTMClassifier(
        input_size=23523,
        hidden_size=256,
        num_layers=2,
        n_out=28
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

tfidf_path = './checkpoint/tfidf_vectorizer_goemotion.pkl'
with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

# # Test text
# test_text = []
# text = input('Input your text: ')
# test_text.append(text)

def Sentiment(text):
    test_text = []
    test_text.append(text)
    test_text_preprocessed = [text.lower() for text in test_text]
    X_test = tfidf.transform(test_text_preprocessed)
    X_test_tensor = torch.FloatTensor(X_test.toarray()).to(device)

    with torch.no_grad():
        outputs1 = model1(X_test_tensor)
        outputs2 = model2(X_test_tensor)

    probabilities_DNN = outputs1
    top5_values_DNN, top5_indices_DNN = torch.topk(probabilities_DNN, k=2)
    top5_values_DNN = top5_values_DNN.cpu()
    top5_indices_DNN = top5_indices_DNN.cpu()

    probabilities_LSTM = outputs2
    top5_values_LSTM, top5_indices_LSTM = torch.topk(probabilities_LSTM, k=2)
    top5_values_LSTM = top5_values_LSTM.cpu()
    top5_indices_LSTM = top5_indices_LSTM.cpu()

    predicted_labels = []
    for i in range(len(test_text)):
        for label_idx, prob in zip(top5_indices_LSTM[i], top5_values_LSTM[i]):
            predicted_labels.append({"label": y.columns[int(label_idx)]})

    return {"predicted": predicted_labels}

print(Sentiment("Fucking shit my engineer friend is so suck"))