import os
import torch
import numpy as np
from sklearn.metrics import classification_report

from src.dataset import load_ag_news, build_vocab, save_vocab, get_dataloaders
from src.model import LSTMClassifier
from src.train import train_one_epoch, validate
from src.evaluate import get_predictions, plot_confusion_matrix


VOCAB_SIZE = 25000
EMBED_DIM = 64
HIDDEN_DIM = 128
MAX_LEN = 50
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.manual_seed(42)
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('Loading AG News dataset...')
train_df, test_df = load_ag_news()
print(f'  Train: {len(train_df)} samples | Test: {len(test_df)} samples')

print('Building vocabulary...')
vocab = build_vocab(train_df['text'].tolist(), max_size=VOCAB_SIZE)
save_vocab(vocab, 'models/vocab.json')
print(f'  Vocab size: {len(vocab)}')

train_loader, test_loader = get_dataloaders(
    train_df, test_df, vocab, MAX_LEN, BATCH_SIZE
)

model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, len(CLASS_NAMES)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print('\nTraining:')
print('-' * 45)
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = validate(
        model, test_loader, criterion, device
    )
    print(
        f'Epoch {epoch+1}/{EPOCHS}  '
        f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
        f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}'
    )

torch.save(model.state_dict(), 'models/lstm_classifier.pt')
print('\nModel saved to models/lstm_classifier.pt')

print('\nTest Results:')
print('-' * 45)
preds, labels = get_predictions(model, test_loader, device)
print(classification_report(labels, preds, target_names=CLASS_NAMES))

plot_confusion_matrix(labels, preds, CLASS_NAMES)
print('Confusion matrix saved to results/confusion_matrix.png')
