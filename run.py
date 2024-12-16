import warnings
import clip
import torch
import os

import argparse
import pytorch_lightning as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import MeldDatasetProcessor
from models import ClassificatorModule, BaseClassificator, ClipClassificator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


warnings.filterwarnings('ignore')


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default='./meld_dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    return parser.parse_args()


def get_dataloaders(preprocess, args):
    dataset_kwargs = dict(
        dataset_path=args.dataset_path,
        image_transform = preprocess,
        text_transform = lambda x: clip.tokenize(x, truncate=True),
        encode_emotion=True,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=4
    )

    train_loader, val_loader, test_loader, label_encoder = (
        MeldDatasetProcessor.create_dataloaders(
            dataset_kwargs = dataset_kwargs,
            dataloader_kwargs = dataloader_kwargs
        )
    )
    return train_loader, val_loader, test_loader, label_encoder

def get_configs():
    model_config = {
        "head_class": BaseClassificator,
        "use_img_encoder": True,
        "use_text_encoder": True,
        "head_hidden_size": 512,
        "output_size": len(MeldDatasetProcessor.EMOTIONS)
    }

    training_config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "scheduler_step_size": 1,
        "scheduler_gamma": 0.8,
    }

    training_config = argparse.Namespace(**training_config)
    return model_config, training_config


def get_optimizer_config(model, training_config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    optimizer_config = {
        "optimizer": optimizer,
        "lr_scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_config.scheduler_step_size,
                gamma=training_config.scheduler_gamma,
            )
        }
    return optimizer_config


def test_cm(training_module, trainer, test_loader, label_encoder):
    y_true = []
    y_pred = []

    training_module.model.to(device)

    for batch in tqdm(test_loader):
        frames = batch['frame'].to(device)
        utterances = batch['utterance'].squeeze(1).to(device)
        emotions = batch['emotion']
        y_true.append(emotions)

        with torch.inference_mode():
            emotions_logits = training_module.model(frames, utterances)
            emotions_pred = torch.argmax(emotions_logits, dim=1).detach()
            y_pred.append(emotions_pred)
    
    y_true = torch.cat(y_true).detach().cpu()
    y_pred = torch.cat(y_pred).detach().cpu()
    test_acc = torch.sum(y_true == y_pred).item() / (len(y_pred) * 1.0)
    print(f"Test accuracy: {test_acc}")

    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.transform(label_encoder.classes_))
    fig, ax = plt.subplots(figsize=(8, 8))  # Define figure size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=label_encoder.classes_)
    
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.close(fig)  # Close to avoid displaying it inline

    trainer.logger.experiment.add_figure("Confusion Matrix", fig, 0)


def main():
    pl.seed_everything(seed=42, workers=True)
    args = _parse_args()

    assert os.path.exists(args.dataset_path), "Dataset path does not exist"
    save_dir = f"./logs/{args.exp_name}"
    assert not os.path.exists(save_dir), "Save directory does not exist"

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()

    model_config, training_config = get_configs()
    model = ClipClassificator(clip_model, **model_config)

    optimizer_config = get_optimizer_config(model, training_config)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Training module
    training_module = ClassificatorModule(model, loss_fn, optimizer_config)

    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(preprocess, args)
    
    learning_rate_callback = pl.callbacks.LearningRateMonitor(
        logging_interval="step",
    )
    tensorboard_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir,
        name=None,
    )

    trainer = pl.Trainer(
        devices=1,
        max_epochs=args.epochs,
        callbacks=[learning_rate_callback],
        logger=tensorboard_logger,
        deterministic=True,
        log_every_n_steps=30,
    )

    trainer.fit(
        training_module,
        train_loader,
        val_loader,
    )

    trainer.test(training_module, test_loader)

    test_cm(training_module, trainer, test_loader, label_encoder)

if __name__ == '__main__':
    main()
