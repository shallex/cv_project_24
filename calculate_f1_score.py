import argparse
import pytorch_lightning as pl
import clip
import torch
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score

from run import get_dataloaders, get_configs
from models import ClassificatorModule, ClipClassificator


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./meld_dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def main():
    args = _parse_args()
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    model_config, _ = get_configs()
    model = ClipClassificator(clip_model, **model_config)
    test_model = ClassificatorModule.load_from_checkpoint(args.checkpoint_path,
                                                 model=model, loss_fn=None, optimizer_config=None)
    test_model.eval()

    _, _, test_loader, _ = get_dataloaders(preprocess, args)

    y_true = []
    y_pred = []

    test_model.to(args.device)

    print('Predicting...')
    for batch in tqdm(test_loader):
        frames = batch['frame'].to(args.device)
        utterances = batch['utterance'].squeeze(1).to(args.device)
        emotions = batch['emotion']
        y_true.append(emotions)

        with torch.no_grad():
            emotions_logits = test_model.model(frames, utterances)
            emotions_pred = torch.argmax(emotions_logits, dim=1).detach()
            y_pred.append(emotions_pred)
    
    y_true = torch.cat(y_true).detach().cpu()
    y_pred = torch.cat(y_pred).detach().cpu()
    acc = torch.sum(y_true == y_pred).item() / (len(y_pred) * 1.0)
    _f1_score = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy: {acc}, F1 score: {_f1_score}')


if __name__ == '__main__':
    main()