'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''


import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from evaluation import test_sonnet

from optimizer import AdamW

TQDM_DISABLE = False


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 기본적으로, 전체 모델을 fine-tuning한다. TODO: 이것은 좋은 생각이 아닌 것 같다.
    for param in self.gpt.parameters():
      param.requires_grad = False

    for name, param in self.gpt.named_parameters():
      if 'A' in name or 'B' in name or 'adapter' in name:
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    ### 완성시켜야 할 빈 코드 블록
    output = self.gpt(input_ids, attention_mask)
    hidden_states = output['last_hidden_state']
    batch_size, seq_length, hidden_dim = hidden_states.shape

    logits = F.linear(hidden_states, self.gpt.word_embedding.weight)

    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=1.0, max_length=128, num_beams=5, repetition_penalty=1.2):
    """
    Beam search와 softmax temperature, repetition penalty를 사용하여 새로운 소넷을 생성한다.
    기존의 top-p 샘플링 방식 대신 beam search를 구현하여 생성 품질을 개선하였다.

    TODO: 지금 이 방법은 기대 이하일 수 있다. 영감을 얻기 위해 Hugging Face의 model.generate(...) 함수를 참고해도 좋겠다.
        여러 시퀀스를 생성하고 beam search를 통해 최적의 시퀀스를 선택하는 것도 좋은 한 가지 방법이다.
        Top-k 샘플링 역시 또 다른 방법이며, 그 외에도 많은 접근법이 있다.
    """
    token_ids = encoding.to(self.get_device())
    device = self.get_device()

    # beam search 초기화
    # 각 beam은 (시퀀스, 점수)의 튜플
    beams = [(token_ids, 0.0)]
    
    for _ in range(max_length):
      all_candidates = []
      
      for seq, score in beams:
        # 최대 길이에 도달했거나 EOS 토큰이 생성된 beam은 완료된 것으로 간주
        if seq.shape[1] >= max_length or seq[0, -1].item() == self.tokenizer.eos_token_id:
          all_candidates.append((seq, score))
          continue
          
        attention_mask = torch.ones_like(seq)
        logits = self.forward(seq, attention_mask)
        logits_last_token = logits[:, -1, :] / temperature # Apply temperature scaling

        # Repetition penalty 적용
        if repetition_penalty != 1.0:
            for token_id in torch.unique(seq[0]):
                if logits_last_token[0, token_id] > 0:
                    logits_last_token[0, token_id] /= repetition_penalty
                else:
                    logits_last_token[0, token_id] *= repetition_penalty

        log_probs = F.log_softmax(logits_last_token, dim=-1)
        top_log_probs, top_indices = torch.topk(log_probs, num_beams, dim=-1)

        for i in range(num_beams):
          next_token = top_indices[:, i].unsqueeze(1)
          log_prob = top_log_probs[:, i].item()
          
          new_seq = torch.cat([seq, next_token], dim=1)
          new_score = score + log_prob
          all_candidates.append((new_seq, new_score))

      # 점수를 기준으로 후보 정렬 (길이 정규화)
      ordered = sorted(all_candidates, key=lambda x: x[1] / x[0].shape[1], reverse=True)
      beams = ordered[:num_beams]

      # 모든 beam이 EOS 토큰으로 끝나면 탐색 종료
      if all(b[0][0, -1].item() == self.tokenizer.eos_token_id for b in beams):
        break

    best_seq, _ = beams[0]
    generated_output = self.tokenizer.decode(best_seq[0].cpu().numpy().tolist())
    
    # 프롬프트 부분 제거
    prompt = self.tokenizer.decode(encoding[0].cpu().numpy().tolist())
    if generated_output.startswith(prompt):
        generated_output = generated_output[len(prompt):]
    
    return best_seq, generated_output


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
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, num_beams=args.num_beams, repetition_penalty=args.repetition_penalty)
      print(f'{batch[1]}{output[1]}\n\n')

    # TODO: 소넷의 작은 테이터셋에서 과적합을 방지하기 위한 종료 조건을 생각하시오.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    prompt_text = batch[1]
    encoding = model.tokenizer(prompt_text, return_tensors='pt', padding=False, truncation=True).to(device)
    _, decoded_output = model.generate(encoding['input_ids'], temperature=args.temperature, num_beams=args.num_beams, repetition_penalty=args.repetition_penalty)
    
    full_sonnet = f'{prompt_text}{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{prompt_text}{decoded_output}\n\n')

  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  # 생성된 소넷 평가
  chrf_score = test_sonnet(args.sonnet_out, args.held_out_sonnet_path)
  print(f"\n생성된 소넷의 CHRF 점수: {chrf_score:.3f}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.0)
  parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search.")
  parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repeating tokens.")

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  generate_submission_sonnets(args)