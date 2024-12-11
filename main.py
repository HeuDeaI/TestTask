import os
import random
from typing import Dict, Union, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Normalize, Compose


class CelebaTripletDataset(Dataset):
    IMAGE_EXT = ".png"  # Extension for the image files
    IMAGE_SIZE = (128, 128)  # Target resize dimensions
    MEAN = [0.5, 0.5, 0.5]  # Normalization mean
    STD = [0.5, 0.5, 0.5]  # Normalization standard deviation

    def __init__(self, images_dir: str, id_mapping_file: str):
        """
        Initialize the dataset with the directory of images and the mapping file.
        """
        self.images_dir = images_dir
        self.id_mapping_file = id_mapping_file
        self.data: Dict[int, List[str]] = {}

        # Validate and load the mapping file
        if not os.path.isfile(id_mapping_file):
            raise ValueError(f"Mapping file '{id_mapping_file}' not found.")
        
        with open(id_mapping_file, "r") as file:
            for line in file:
                image_id, person_id = line.split()
                person_id = int(person_id)
                if person_id not in self.data:
                    self.data[person_id] = []
                self.data[person_id].append(image_id.replace(".jpg", ""))
        
        # Validate that we have at least two images per person
        self.data = {pid: img_ids for pid, img_ids in self.data.items() if len(img_ids) >= 2}
        if not self.data:
            raise ValueError("No valid entries in the mapping file. Ensure each person has at least two images.")

        self.person_ids = list(self.data.keys())  # Precompute the list of person IDs

    def __len__(self) -> int:
        return len(self.person_ids)

    def __getitem__(self, index: int) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Get a triplet: base anchor, positive anchor, and negative anchor.
        """
        if index < 0 or index >= len(self.person_ids):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self.person_ids)}.")

        # Get the person ID for the given index
        person_id = self.person_ids[index]
        positive_samples = random.sample(self.data[person_id], 2)
        base_anch, pos_anch = positive_samples

        # Select a negative person ID
        neg_person_id = random.choice([pid for pid in self.person_ids if pid != person_id])
        neg_anch = random.choice(self.data[neg_person_id])

        return {
            "person_id": person_id,
            "base_anchor": self._load_image_as_tensor(base_anch),
            "pos_anchor": self._load_image_as_tensor(pos_anch),
            "neg_anchor": self._load_image_as_tensor(neg_anch),
        }

    def _load_image_as_tensor(self, image_id: str) -> torch.Tensor:
        """
        Load an image and transform it into a normalized tensor.
        """
        image_path = os.path.join(self.images_dir, f"{image_id}{self.IMAGE_EXT}")
        if not os.path.isfile(image_path):
            raise ValueError(f"Image file '{image_path}' not found.")
        
        try:
            image = Image.open(image_path).convert("RGB")
            transform = Compose([
                Resize(self.IMAGE_SIZE),
                ToTensor(),
                Normalize(mean=self.MEAN, std=self.STD),
            ])
            return transform(image)
        except Exception as e:
            raise ValueError(f"Error processing image '{image_path}': {e}")

if __name__ == "__main__":
    # For testing and debugging
    try:
        dataset = CelebaTripletDataset(images_dir="img_align_celeba_png", id_mapping_file="identity_CelebA.txt")
        sample_data = dataset[random.randint(0, len(dataset) - 1)]
        print("Sample Data:", sample_data)
    except Exception as e:
        print(f"Error: {e}")
