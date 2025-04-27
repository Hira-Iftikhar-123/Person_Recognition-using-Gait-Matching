import torch
import numpy as np
import joblib
from models.classifier import GaitClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_labels = np.load('train_labels.npy')
val_labels = np.load('val_labels.npy')
val_features_kce = np.load('val_features_kce.npy')
input_dim = int(np.load('input_dim.npy'))

classifier = GaitClassifier(input_dim=input_dim, num_classes=len(set(train_labels))).to(device)
classifier.load_state_dict(torch.load('model_checkpoint.pth'))
classifier.eval()

X_test = torch.tensor(val_features_kce, dtype=torch.float32).to(device)
y_test = torch.tensor(val_labels, dtype=torch.long).to(device)

with torch.no_grad():
    outputs = classifier(X_test)
    _, preds = torch.max(outputs, 1)
    acc = (preds == y_test).sum().item() / y_test.size(0)

print(f'Test Accuracy: {acc * 100:.2f}%')

print("\nTesting One-vs-All SVM...")
svm_classifier = joblib.load('svm_model.pkl')

svm_preds = svm_classifier.predict(val_features_kce)

svm_acc = (svm_preds == val_labels).sum() / len(val_labels)
print(f'SVM Test Accuracy: {svm_acc * 100:.2f}%')
