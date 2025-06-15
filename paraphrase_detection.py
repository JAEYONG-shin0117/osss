'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu --use_augmentation`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from numpy.core.multiarray import _reconstruct

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)

from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW
from transformers import MarianMTModel, MarianTokenizer

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def back_translate(text, tokenizer_pivot, model_pivot, tokenizer_back, model_back, device):
    try:
        if not text or len(text.strip()) < 3:
            return text
        if len(text) > 500:
            text = text[:500]

        inputs = tokenizer_pivot.prepare_seq2seq_batch(
            [text], return_tensors="pt", max_length=128, truncation=True)
        if device.type == 'cuda':
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            translated = model_pivot.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True)

        pivot_text = tokenizer_pivot.decode(translated[0], skip_special_tokens=True)

        inputs_back = tokenizer_back.prepare_seq2seq_batch(
            [pivot_text], return_tensors="pt", max_length=128, truncation=True)
        if device.type == 'cuda':
            inputs_back = {k: v.to(device) for k, v in inputs_back.items()}

        with torch.no_grad():
            back_translated = model_back.generate(
                **inputs_back, max_length=128, num_beams=4, early_stopping=True)

        result = tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
        if result.lower().strip() == text.lower().strip():
            return None
        return result
    except Exception as e:
        print(f"Back translation failed for text: {text[:50]}... Error: {str(e)}")
        return None

def augment_data_with_back_translation(train_data, tokenizer_pivot, model_pivot, 
                                     tokenizer_back, model_back, device, 
                                     augment_ratio=0.5, max_augment=1000):
    augmented_data = []
    augment_count = min(int(len(train_data) * augment_ratio), max_augment)

    if device.type == 'cuda':
        model_pivot.to(device)
        model_back.to(device)

    model_pivot.eval()
    model_back.eval()

    print(f"Starting back translation augmentation for {augment_count} samples...")
    selected_indices = random.sample(range(len(train_data)), augment_count)
    successful_augmentations = 0

    for idx in tqdm(selected_indices, desc="Back Translation"):
        item = train_data[idx]
        q1, q2, label = item['question1'], item['question2'], item['label']
        aug_q1 = back_translate(q1, tokenizer_pivot, model_pivot, tokenizer_back, model_back, device)
        aug_q2 = back_translate(q2, tokenizer_pivot, model_pivot, tokenizer_back, model_back, device)

        if aug_q1 and aug_q2:
            augmented_data.append({'question1': aug_q1, 'question2': aug_q2, 'label': label})
            successful_augmentations += 1

    print(f"Successfully augmented {successful_augmentations} samples out of {augment_count} attempts")

    if device.type == 'cuda':
        model_pivot.cpu()
        model_back.cpu()
        torch.cuda.empty_cache()

    return augmented_data

class ParaphraseGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids, attention_mask)
    hidden_states = outputs['last_hidden_state']
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.sum(mask_expanded, dim=1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_hidden = sum_hidden / sum_mask
    logits = self.paraphrase_detection_head(mean_hidden)
    return logits

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def train(args):
  args = add_arguments(args)
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  tokenizer_pivot = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
  model_pivot = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
  tokenizer_back = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
  model_back = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')

  if args.use_augmentation:
    augmented_data = augment_data_with_back_translation(
      para_train_data, tokenizer_pivot, model_pivot,
      tokenizer_back, model_back, device,
      augment_ratio=args.augment_ratio,
      max_augment=args.max_augment_samples)
    print(f"Original training data: {len(para_train_data)}")
    para_train_data.extend(augmented_data)
    print(f"Augmented training data: {len(para_train_data)}")

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)
  model = ParaphraseGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
  best_dev_acc = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    train_loss /= len(para_train_dataloader)
    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

@torch.no_grad()
def test(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  with torch.serialization.safe_globals([argparse.Namespace, _reconstruct]):
    saved = torch.load(args.filepath)
  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--use_augmentation", action='store_true')
  parser.add_argument("--augment_ratio", type=float, default=0.3)
  parser.add_argument("--max_augment_samples", type=int, default=500)
  args = parser.parse_args()
  return args

def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d = 768; args.l = 12; args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024; args.l = 24; args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280; args.l = 36; args.num_heads = 20
  else:
    raise Exception(f"{args.model_size} is not supported.")
  return args

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'
  seed_everything(args.seed)
  train(args)
  test(args)