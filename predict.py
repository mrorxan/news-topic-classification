import torch

from src.dataset import load_vocab, encode_text
from src.model import LSTMClassifier


CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']

vocab = load_vocab('models/vocab.json')

model = LSTMClassifier(len(vocab), embed_dim=64, hidden_dim=128, num_classes=4)
model.load_state_dict(torch.load('models/lstm_classifier.pt', map_location='cpu'))
model.eval()

samples = [
    "Stock markets rallied today as investors reacted to strong earnings reports from major tech companies",
    "The national football team secured a dramatic victory in the World Cup qualifier match",
    "NASA scientists announced the discovery of a new exoplanet in a nearby star system",
    "European Union leaders gathered in Brussels to discuss trade agreements and economic policy",
    "The tennis champion won her third consecutive Grand Slam title after a grueling final",
    "Researchers developed a new algorithm that improves battery efficiency in electric vehicles",
]

print('News Topic Classification')
print('=' * 60)

for text in samples:
    encoded = encode_text(text, vocab, max_len=50)
    input_tensor = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    print(f'\n  Text: {text[:80]}...')
    print(f'  Predicted: {CLASS_NAMES[pred]}')
