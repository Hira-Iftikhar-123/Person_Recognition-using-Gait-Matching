import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import joblib
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.dataset_loader import GaitSilhouetteDataset
from models.resnet_extractor import ResNet101Extractor
from models.classifier import GaitClassifier
from utils.feature_selection_kce import kce_feature_selection
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          
    transforms.RandomRotation(5),               
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = GaitSilhouetteDataset(root_dir='datasets/GaitDatasetA-silh/', transform=transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

resnet_extractor = ResNet101Extractor().to(device)
resnet_extractor.eval()

def extract_and_aggregate_features(loader):
    feature_dict = defaultdict(list)
    label_dict = defaultdict(list)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = resnet_extractor(images).cpu().numpy()
            for f, l in zip(features, labels.numpy()):
                feature_dict[l].append(f)
                label_dict[l].append(l)

    final_features = []
    final_labels = []
    for label in feature_dict.keys():
        aggregated_feature = np.mean(feature_dict[label], axis=0)
        final_features.append(aggregated_feature)
        final_labels.append(label)

    return np.array(final_features), np.array(final_labels)

train_features, train_labels = extract_and_aggregate_features(train_loader)
val_features, val_labels = extract_and_aggregate_features(val_loader)

train_features_kce, selected_indices = kce_feature_selection(train_features)
val_features_kce = val_features[:, selected_indices]

classifier = GaitClassifier(input_dim=train_features_kce.shape[1], num_classes=len(set(train_labels))).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_train = torch.tensor(train_features_kce, dtype=torch.float32).to(device)
y_train = torch.tensor(train_labels, dtype=torch.long).to(device)

for epoch in range(50):
    classifier.train()
    optimizer.zero_grad()
    outputs = classifier(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    classifier.eval()
    with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_train).sum().item()
        total = y_train.size(0)
        accuracy = correct/total

    print(f'Epoch [{epoch+1}/20], Loss: {loss.item():.4f}, Train Accuracy: {accuracy * 100:.2f}%')

np.save('train_labels.npy', train_labels)
np.save('val_labels.npy', val_labels)
np.save('val_features_kce.npy', val_features_kce)
np.save('input_dim.npy', train_features_kce.shape[1])
np.save('selected_indices.npy', selected_indices)
torch.save(classifier.state_dict(), 'model_checkpoint.pth')

print("Training One-vs-All SVM...")
svm_classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
svm_classifier.fit(train_features_kce, train_labels)

joblib.dump(svm_classifier, 'svm_model.pkl')
print("SVM model saved as svm_model.pkl")
