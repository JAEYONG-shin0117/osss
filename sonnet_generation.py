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
  def generate(
      self,
      encoding,
      beam_width: int = 3,
      max_length: int = 128,
      length_penalty: float = 0.7
  ):
    """
    Beam Search를 통해 최적의 시퀀스를 생성한다.

    Args:
        encoding: 이미 토크나이즈되어 tensor 형태인 입력 (예: tokenizer(..., return_tensors='pt'))
        beam_width: 빔 폭 (beam size)
        max_length: 최대 생성 길이 (토큰 수)
        length_penalty: 길이 패널티 계수 (짧은 시퀀스를 과도히 선호하지 않도록)
    Returns:
        best_sequence_ids: (tensor) 최종 선택된 토큰 ID 시퀀스
        best_decoded: (str) 디코딩된 문자열
    """
    device = self.get_device()
    # encoding['input_ids']는 (1, seq_len) 형태가 되도록 가정
    input_ids = encoding['input_ids'].to(device)           # (1, L0)
    attention_mask = torch.ones_like(input_ids).to(device)  # (1, L0)

    # ------------------------------------------------------------
    # 1) 초기 빔 상태 설정
    # ------------------------------------------------------------
    # beam_list: list of dict, each dict = {
    #     "token_ids": Tensor (1, cur_len),
    #     "attention_mask": Tensor (1, cur_len),
    #     "cum_logprob": float  (누적 로그 확률)
    # }
    beam_list = [{
        "token_ids": input_ids,        # (1, L0)
        "attention_mask": attention_mask,  # (1, L0)
        "cum_logprob": 0.0
    }]

    # EOS를 만난 “완성 빔(finished beams)”을 저장할 리스트
    finished_beams = []

    # ------------------------------------------------------------
    # 2) 매 스텝마다 빔 확장 (확장 후 다시 beam_width개 선별)
    # ------------------------------------------------------------
    for _ in range(max_length):
        all_candidates = []

        # 현재 살아있는 빔 각각에 대해 확장
        for beam in beam_list:
            seq_ids = beam["token_ids"]           # (1, cur_len)
            seq_mask = beam["attention_mask"]     # (1, cur_len)
            cum_logprob = beam["cum_logprob"]     # float

            # 마지막 토큰에 대한 logits 계산
            logits_sequence = self.forward(seq_ids, seq_mask)   # (1, cur_len, vocab_size)
            logits_last = logits_sequence[:, -1, :]             # (1, vocab_size)
            log_probs = torch.log_softmax(logits_last, dim=-1)  # (1, vocab_size)

            # top_k개 토큰을 뽑아야 하나, 일단 빔 폭만큼 뽑아본다
            topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_width, dim=-1)  # 둘 다 (1, beam_width)

            topk_log_probs = topk_log_probs.squeeze(0)    # (beam_width,)
            topk_indices = topk_indices.squeeze(0)        # (beam_width,)

            # 각 후보 토큰마다 새롭게 빔 후보를 만들어 all_candidates에 추가
            for i in range(beam_width):
                token_id = topk_indices[i].unsqueeze(0).unsqueeze(0)  # (1,1)
                token_logprob = topk_log_probs[i].item()
                new_cum_logprob = cum_logprob + token_logprob

                # 이전 시퀀스에 토큰을 붙이고, 마스크도 확장
                new_seq_ids = torch.cat([seq_ids, token_id], dim=1)  # (1, cur_len+1)
                new_seq_mask = torch.cat([seq_mask, torch.ones((1,1), dtype=torch.int64).to(device)], dim=1)

                candidate = {
                    "token_ids": new_seq_ids,
                    "attention_mask": new_seq_mask,
                    "cum_logprob": new_cum_logprob
                }
                all_candidates.append(candidate)

        # all_candidates에는 beam_width * (현재 빔 개수)개의 후보가 들어있다.
        # 이 중 상위 beam_width개만 선택
        # 먼저 각 후보의 “점수(score)”를 계산: (누적로그확률 / length_penalty)
        # length_penalty를 적용하여, 지나치게 짧은 시퀀스가 선택되지 않도록 보정
        # 점수 = cum_logprob / (seq_len ** length_penalty)
        scored_candidates = []
        for cand in all_candidates:
            seq_len = cand["token_ids"].shape[1]
            score = cand["cum_logprob"] / ( (seq_len ** length_penalty) )
            scored_candidates.append((score, cand))

        # 점수 순으로 내림차순 정렬
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 베스트 beam_width개만 beam_list로 유지
        new_beam_list = []
        for idx in range(min(beam_width, len(scored_candidates))):
            _, best_cand = scored_candidates[idx]
            new_beam_list.append(best_cand)

        beam_list = new_beam_list

        # --------------------------------------------------------
        # 3) EOS 토큰이 포함된 빔은 finished_beams로 옮겨야 함
        # --------------------------------------------------------
        still_alive_beams = []
        for beam in beam_list:
            last_token_id = beam["token_ids"][0, -1].item()
            if last_token_id == self.tokenizer.eos_token_id:
                # EOS를 만난 빔 → finished_beams에 저장 (score를 함께 저장)
                seq_len = beam["token_ids"].shape[1]
                score = beam["cum_logprob"] / ( (seq_len ** length_penalty) )
                finished_beams.append((score, beam))
            else:
                still_alive_beams.append(beam)

        beam_list = still_alive_beams

        # 만약 살아 있는 빔이 하나도 없다면 반복 종료
        if len(beam_list) == 0:
            break

    # ------------------------------------------------------------
    # 4) “완성된 빔”이 없다면, 아직 살아남은 빔 중 최고 점수를 고른다
    # ------------------------------------------------------------
    if len(finished_beams) == 0:
        # beam_list에 남아있는 것들을 finished로 간주 (EOS는 없지만)
        for beam in beam_list:
            seq_len = beam["token_ids"].shape[1]
            score = beam["cum_logprob"] / ( (seq_len ** length_penalty) )
            finished_beams.append((score, beam))

    # ------------------------------------------------------------
    # 5) finished_beams 중 최고 점수를 선택하여 결과 반환
    # ------------------------------------------------------------
    finished_beams.sort(key=lambda x: x[0], reverse=True)
    best_score, best_beam = finished_beams[0]
    best_sequence_ids = best_beam["token_ids"]  # (1, final_len)

    # 디코딩할 때, 처음에 붙어있던 프롬프트(입력부분)를 제외하려면 슬라이싱 가능
    # 예: best_sequence_ids[0, input_len:].tolist()
    decoded = self.tokenizer.decode(best_sequence_ids[0].cpu().tolist(), skip_special_tokens=True)

    return best_sequence_ids, decoded
     

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
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
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

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!# held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

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