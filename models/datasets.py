import os
import shutil
import kagglehub
import torch
import torchvision
from torchvision.datasets import ImageFolder

class VegetablesDataset(ImageFolder):
    """
    Custom Dataset class for the Kaggle Vegetable Image Dataset.
    Behaves exactly like standard torchvision datasets (CIFAR10, etc.).
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        # Define where the data will live inside your args.data_root
        self.dataset_root = os.path.join(root, "vegetables")
        
        if download:
            self.download_dataset()

        # Decide which split folder to use based on the `train` flag
        # The Kaggle dataset contains 'train', 'test', and 'validation' folders
        split_folder = "train" if train else "validation" 
        target_dir = os.path.join(self.dataset_root, "Vegetable Images", split_folder)

        if not os.path.exists(target_dir):
            raise RuntimeError(f"Dataset not found at {target_dir}. Please set download=True.")

        # Initialize the parent ImageFolder class with the exact split directory
        super().__init__(
            root=target_dir,
            transform=transform,
            target_transform=target_transform
        )

    def download_dataset(self):
        """Downloads the dataset via kagglehub and moves it to the target root."""
        # Check if it already exists to avoid re-downloading/copying
        if os.path.exists(os.path.join(self.dataset_root, "Vegetable Images", "train")):
            print("Files already downloaded and verified.")
            return

        print("Downloading Vegetable dataset via kagglehub...")
        # kagglehub downloads to a hidden cache folder by default
        cache_path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")

        print(f"Moving dataset from cache to your data root: {self.dataset_root}...")
        os.makedirs(self.dataset_root, exist_ok=True)
        
        # Copy the files from the cache to your intended root directory
        shutil.copytree(cache_path, self.dataset_root, dirs_exist_ok=True)
        print("Download and setup complete!")