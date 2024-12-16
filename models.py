import torch
import torch.nn as nn
import pytorch_lightning as pl


class BaseClassificator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseClassificator, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    

class ClipClassificator(nn.Module):
    def __init__(self, clip, head_class, use_img_encoder: bool, use_text_encoder: bool, head_hidden_size: int, output_size: int):
        """
        Args:
            clip: nn.Module
            head_class: Callable
            use_img_encoder: bool
            use_text_encoder: bool
            output_size: int
        """
        super(ClipClassificator, self).__init__()
        self.clip = clip
        head_input_size = 0
        if use_img_encoder:
            head_input_size += self.clip.ln_final.normalized_shape[0]
        if use_text_encoder:
            head_input_size += self.clip.ln_final.normalized_shape[0]
        self.head = head_class(head_input_size, head_hidden_size, output_size)
        self.use_img_encoder = use_img_encoder
        self.use_text_encoder = use_text_encoder
    
    def forward(self, image, text):
        if self.use_img_encoder:
            features = self.clip.encode_image(image).float()
        if self.use_text_encoder:
            text_features = self.clip.encode_text(text).float()
            if self.use_img_encoder:
                features = torch.cat((features, text_features), dim=1)
            else:
                features = text_features

        x = self.head(features)
        return x


class ClassificatorModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer_config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
    
    def training_step(self, batch, batch_idx):
        frames = batch['frame']
        utterances = batch['utterance'].squeeze(1)
        emotions = batch['emotion']

        emotions_logits = self.model(frames, utterances)
        loss = self.loss_fn(emotions_logits, emotions)
        self.log('train/loss', loss)
        emotions_pred = torch.argmax(emotions_logits, dim=1).detach()
        acc = torch.sum(emotions == emotions_pred).item() / (len(emotions) * 1.0)
        self.log('train/accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        frames = batch['frame']
        utterances = batch['utterance'].squeeze(1)
        emotions = batch['emotion']

        emotions_logits = self.model(frames, utterances)
        loss = self.loss_fn(emotions_logits, emotions)
        self.log('val/loss', loss)
        emotions_pred = torch.argmax(emotions_logits, dim=1).detach()
        acc = torch.sum(emotions == emotions_pred).item() / (len(emotions) * 1.0)
        self.log('val/accuracy', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        frames = batch['frame']
        utterances = batch['utterance'].squeeze(1)
        emotions = batch['emotion']

        emotions_logits = self.model(frames, utterances)
        emotions_pred = torch.argmax(emotions_logits, dim=1).detach()
        acc = torch.sum(emotions == emotions_pred).item() / (len(emotions) * 1.0)

        self.log('test/accuracy', acc)

    def configure_optimizers(self):
        return self.optimizer_config
