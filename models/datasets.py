import os
import shutil
import kagglehub
import torch
from torchvision.datasets import ImageFolder

class VegetablesDataset(ImageFolder):
    """
    Custom Dataset class for the Kaggle Vegetable Image Dataset.
    Behaves exactly like standard torchvision datasets (CIFAR10, etc.).
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, classes=None):
        self.dataset_root = os.path.join(root, "vegetables")
        
        # Store the requested classes BEFORE calling super().__init__
        self.target_classes = classes 
        
        if download:
            self.download_dataset()

        # Decide which split folder to use based on the `train` flag
        split_folder = "train" if train else "validation" 
        target_dir = os.path.join(self.dataset_root, "Vegetable Images", split_folder)

        if not os.path.exists(target_dir):
            raise RuntimeError(f"Dataset not found at {target_dir}. Please set download=True.")

        # Initialize the parent ImageFolder class
        super().__init__(
            root=target_dir,
            transform=transform,
            target_transform=target_transform
        )

    def find_classes(self, directory: str):
        """
        Overrides the default ImageFolder behavior to filter out unwanted classes.
        """
        # Fetch all available classes using the parent method
        all_classes, _ = super().find_classes(directory)

        # If no specific classes were requested, return everything
        if self.target_classes is None:
            all_class_to_idx = {cls_name: i for i, cls_name in enumerate(all_classes)}
            return all_classes, all_class_to_idx

        # Filter to keep only the requested classes that actually exist
        filtered_classes = [c for c in all_classes if c in self.target_classes]
        
        if not filtered_classes:
            raise ValueError(f"None of the target classes {self.target_classes} were found in {directory}.")

        # Re-map class indices so they are contiguous (0, 1, 2...)
        filtered_class_to_idx = {cls_name: i for i, cls_name in enumerate(filtered_classes)}

        return filtered_classes, filtered_class_to_idx

    def download_dataset(self):
        """Downloads the dataset via kagglehub and moves it to the target root."""
        if os.path.exists(os.path.join(self.dataset_root, "Vegetable Images", "train")):
            print("Files already downloaded and verified.")
            return

        print("Downloading Vegetable dataset via kagglehub...")
        cache_path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")

        print(f"Moving dataset from cache to your data root: {self.dataset_root}...")
        os.makedirs(self.dataset_root, exist_ok=True)
        
        shutil.copytree(cache_path, self.dataset_root, dirs_exist_ok=True)
        print("Download and setup complete!")
