import os
import urllib.request
import gzip
import tarfile
import pickle
import numpy as np
from .core import Dataset, DataLoader

class MNISTDataset(Dataset):
    url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"

    def __init__(self, root='./mnist_data', train=True):
        self.root = root
        self.train = train
        self.filepath = os.path.join(root, 'mnist.pkl.gz')
        os.makedirs(root, exist_ok=True)

        if not os.path.exists(self.filepath):
            print(f"Downloading MNIST dataset from {self.url}")
            urllib.request.urlretrieve(self.url, self.filepath)
        else:
            print("MNIST dataset already downloaded.")

        with gzip.open(self.filepath, 'rb') as f:
            train_set, val_set, test_set = pickle.load(f, encoding='latin1')

        if train:
            data, targets = train_set
        else:
            data, targets = test_set

        self.images = data.reshape(-1, 28, 28).astype(np.float32)
        self.labels = targets.astype(np.int64)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class CIFAR100Dataset(Dataset):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

    def __init__(self, root='./cifar100_data', train=True):
        self.root = root
        self.train = train
        self.dataset_folder = os.path.join(root, 'cifar-100-python')
        self.filepath = os.path.join(root, 'cifar-100-python.tar.gz')

        os.makedirs(root, exist_ok=True)

        if not os.path.exists(self.dataset_folder):
            print(f"Downloading CIFAR-100 dataset from {self.url} ...")
            urllib.request.urlretrieve(self.url, self.filepath)
            print("Extracting...")
            with tarfile.open(self.filepath, 'r:gz') as tar:
                tar.extractall(path=root)
            print("Extraction done.")
        else:
            print("CIFAR-100 dataset already exists.")

        if train:
            file = os.path.join(self.dataset_folder, 'train')
        else:
            file = os.path.join(self.dataset_folder, 'test')

        with open(file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        data = data_dict[b'data']
        labels = data_dict[b'fine_labels']

        self.images = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.labels = np.array(labels, dtype=np.int64)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# 测试
if __name__ == "__main__":
    """
    print("Testing MNISTDataset and DataLoader:")
    mnist_train = MNISTDataset(train=True)
    print(f"Train samples: {len(mnist_train)}")
    print("Sample image shape:", mnist_train[0][0].shape)
    print("Sample label:", mnist_train[0][1])

    mnist_loader = DataLoader(mnist_train, batch_size=16, shuffle=True)
    for i, (batch_images, batch_labels) in enumerate(mnist_loader):
        print(f"Batch {i} images shape: {batch_images.shape}")
        print(f"Batch {i} labels: {batch_labels}")
        if i >= 1:
            break
    """

    dataset = CIFAR100Dataset(train=True)
    print(f"Number of training samples: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_imgs, batch_labels in loader:
        print("Batch images shape:", batch_imgs.shape)  # (8, 3, 32, 32)
        print("Batch labels:", batch_labels)
        break