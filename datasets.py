# !/usr/bin/env python3

"""
이 파일은 Quora의 Paraphrase Detection을 위한 Dataset 클래스를 포함한다. 추가 데이터 소스로 훈련시키거나
Quora 데이터셋의 처리 방식(예: 데이터 증강 등)을 변경하려는 경우 이 파일을 수정할 수 있다.
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

import random
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, n=1):
    """
    문장에서 n개의 단어를 유의어로 바꿉니다.
    """
    words = text.split()
    eligible_words = [word for word in words if get_synonyms(word)]
    if not eligible_words:
        return text  # 바꿀 수 있는 단어가 없으면 원문 반환

    random.shuffle(eligible_words)
    num_replaced = 0

    for word in eligible_words:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            words = [synonym if w == word else w for w in words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return " ".join(words)


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  """
  데이터 증강 파인튜닝을 위한 코드 수정

  """
  def __getitem__(self, idx):
    # 원본 데이터 추출
    row = self.dataset[idx]
    text1, text2, label, _ = row  # 또는 row[0], row[1], row[2]
    text1 = str(text1)
    text2 = str(text2)
    label = int(label)



    # 데이터 증강 (Synonym Replacement) -------------------
    # 학습 데이터를 더욱 일반화된 형태로 만들기 위해 유의어 치환(synonym replacement)방법
    # 효과: 모델이 다양한 문장 표현을 학습하게 하여 과적합을 줄이고 일반화 성능 향상 가능
    if self.p.augment and random.random() < 0.3:
        text1 = synonym_replacement(text1, n=2)
        text2 = synonym_replacement(text2, n=2)

    # 토큰화
    inputs = self.tokenizer(
        text1,
        text2,
        truncation=True,
        padding='max_length',
        max_length=self.p.max_len,
        return_tensors='pt'
    )

    item = {key: val.squeeze(0) for key, val in inputs.items()}
    item['labels'] = torch.tensor(label)
    
    return (text1, text2, label, row[3]) if len(row) > 3 else (text1, text2, label, str(idx))


  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    labels = torch.LongTensor([x[2] for x in all_data])
    # labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    # labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                         (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return {
        'input_ids': self.input_ids[idx],
        'attention_mask': self.attention_mask[idx],
        'labels': self.labels[idx]
    }


  def collate_fn(self, all_data):
    input_ids = torch.stack([x['input_ids'] for x in all_data])
    attention_mask = torch.stack([x['attention_mask'] for x in all_data])
    labels = torch.stack([x['labels'] for x in all_data])

    return {
        'token_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
