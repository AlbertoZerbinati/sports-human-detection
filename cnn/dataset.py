# Alberto Zerbinati

import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PeopleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load positive examples (label 1)
        self.positive_files = [
            os.path.join(root_dir, "positive", f)
            for f in os.listdir(os.path.join(root_dir, "positive"))
            if os.path.isfile(os.path.join(root_dir, "positive", f))
        ]

        # Load negative examples (label 0)
        self.negative_files = [
            os.path.join(root_dir, "negative", f)
            for f in os.listdir(os.path.join(root_dir, "negative"))
            if os.path.isfile(os.path.join(root_dir, "negative", f))
        ]

        # Combine both
        self.all_files = [(f, 1) for f in self.positive_files] + [
            (f, 0) for f in self.negative_files
        ]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.all_files[idx]
        image = Image.open(img_path).convert("RGB")  # down to 3 channesl also for PNGs

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    )

    # Initialize train/valid/test datasets and dataloaders
    train_path = "data/dataset/train"
    valid_path = "data/dataset/valid"
    test_path = "data/dataset/test"

    train_dataset = PeopleDataset(train_path, transform=transform)
    valid_dataset = PeopleDataset(valid_path, transform=transform)
    test_dataset = PeopleDataset(test_path, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    sample_image, label = train_dataset[0]

    print(sample_image.shape)
    print(label)
