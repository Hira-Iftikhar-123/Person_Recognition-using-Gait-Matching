import os
import cv2
from torch.utils.data import Dataset
from PIL import Image  

class GaitSilhouetteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        person_folders = sorted(os.listdir(root_dir))
        person_folders = [f for f in person_folders if os.path.isdir(os.path.join(root_dir, f))]

        for label, person_folder in enumerate(person_folders):
            person_path = os.path.join(root_dir, person_folder)
            sequence_folders = sorted(os.listdir(person_path))
            for seq_folder in sequence_folders:
                seq_path = os.path.join(person_path, seq_folder)
                if os.path.isdir(seq_path):
                    images = os.listdir(seq_path)
                    for img in images:
                        if img.endswith('.png') or img.endswith('.jpg'):
                            img_path = os.path.join(seq_path, img)
                            self.samples.append(img_path)
                            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        image = Image.fromarray(image)  

        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label