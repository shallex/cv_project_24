import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class MeldDatasetProcessor(Dataset):
    SPLITS = ['train', 'dev', 'test']
    EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    def __init__(self, dataset_path, split, encode_emotion=True, image_transform=None, text_transform=None):

        self.split = split
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.images_dir = os.path.join(dataset_path, split)
        self.mapping = pd.read_csv(os.path.join(dataset_path, f"{split}.csv"))
        self.emotion_encoder = self._initialize_emotion_encoder()
        self.encode_emotion = encode_emotion

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        item_data = self.mapping.iloc[idx]
        image_name = item_data["Image_Name"] 
        image = self._process_image(image_name)

        if self.encode_emotion:
            emotion = self.emotion_encoder.transform([item_data["Emotion"]])[0]
        else:
            emotion = self._process_text(item_data["Emotion"])

        return {
            'frame': image,
            'utterance': self._process_text(item_data["Utterance"]),
            'emotion': emotion
        }

    def _initialize_emotion_encoder(self):
        encoder = LabelEncoder()
        return encoder.fit(self.EMOTIONS)

    def _process_image(self, image_name):
        image = Image.open(os.path.join(self.images_dir, image_name))
        if self.image_transform:
            image = self.image_transform(image)
    
        return image

    def _process_text(self, text):        
        if self.text_transform:
            text = self.text_transform(text)
    
        return text

    @classmethod
    def create_dataloaders(cls, dataset_kwargs, dataloader_kwargs):
            datasets = {
                split: cls(**dataset_kwargs, split=split)
                for split in cls.SPLITS
            }

            dataloaders = {
                split: DataLoader(
                    dataset,
                    shuffle=(split == 'train'),
                    pin_memory=True,
                    **dataloader_kwargs
                )
                for split, dataset in datasets.items()
            }

            return (
                dataloaders['train'],
                dataloaders['dev'],
                dataloaders['test'],
                datasets['train'].emotion_encoder
            )