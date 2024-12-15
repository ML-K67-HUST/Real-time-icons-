import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from ANN import ANN

model_path = './model_checkpoint.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANN(n_in=16909, n_out=90).to(device)

checkpoint = torch.load(model_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

tfidf_path = './tfidf_vectorizer.pkl'
mlb_path = './multilable.pkl'
with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)
with open(mlb_path, 'rb') as f:
    mlb = pickle.load(f)

# Test text
test_text = [
    "JOHOR BAHRU: By mid-2025, Singaporeans and other foreign travellers may be able to clear Johor land checkpoints with just QR codes without needing to show their passports. A senior official of the Johor state government Lee Ting Han said on Wednesday (Dec 11) that Malaysian authorities are aiming for QR code immigration clearance to be expanded to Singaporeans and other foreign passport holders by the middle of next year."
]

test_text_preprocessed = [text.lower() for text in test_text]
X_test = tfidf.transform(test_text_preprocessed)
X_test_tensor = torch.FloatTensor(X_test.toarray()).to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
probabilities = outputs
top5_values, top5_indices = torch.topk(probabilities, k=5)

top5_values = top5_values.cpu()
top5_indices = top5_indices.cpu()

print("\nTop 5 Predicted Labels and Probabilities:")
for i in range(len(test_text)):
    for label_idx, prob in zip(top5_indices[i], top5_values[i]):
        label = mlb.classes_[label_idx]
        print(f"  {label}: {prob:.4f}")