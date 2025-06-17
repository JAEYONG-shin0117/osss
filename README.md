# 자연어처리 2025-1 지정주제 기말 프로젝트: GPT-2 구축

팀원 : 2022113591 신재용 2020112036 김상현 2022113596 오유나

## 환경 설정

#### 파이썬 설치
* Python 3.8 이상을 설치한다.

#### 가상환경 생성 및 패키지 설치

```bash
# 가상환경 생성
python -m venv nlp_env

# 가상환경 활성화
# Windows:
nlp_env\Scripts\activate
# macOS/Linux:
source nlp_env/bin/activate

# 필수 패키지 설치
pip install -r requirements.txt
```

#### Google Colab에서 실행하기 (권장)
- Colab에서는 `!`(느낌표)로 리눅스 명령어, `%cd`로 디렉토리 이동을 실행합니다.
- [런타임] → [런타임 유형 변경] → [하드웨어 가속기: GPU]로 꼭 GPU를 선택하세요.
- 데이터 파일이 필요하다면 직접 업로드하거나, 구글 드라이브를 마운트해서 사용하세요.
- Colab 세션은 일정 시간이 지나면 초기화될 수 있으니, 중간 결과는 저장해두세요.
- GPU 메모리 부족 시 `--batch_size` 값을 줄여서 실행하세요.


## 실행 방법 (Google Colab 기준)

아래 명령어를 Colab 셀에 순서대로 입력하면 모든 실험을 바로 실행할 수 있습니다.

```python
# 1. 저장소 클론
!git clone https://github.com/JAEYONG-shin0117/osss.git

# 2. 작업 디렉토리 이동
%cd osss

# 3. 필수 패키지 설치
!pip install -r requirements.txt

# 4. Paraphrase Detection 실행
!python paraphrase_detection.py --use_gpu

# 5. Sonnet Generation 실행
!python sonnet_generation.py --use_gpu

# 6. Sentiment Classification 실행
!python classifier.py --use_gpu
```


## 데이터 파일 구조

```
data/
├── quora-train.csv          # 패러프레이즈 탐지 학습 데이터
├── quora-dev.csv            # 패러프레이즈 탐지 검증 데이터
├── quora-test-student.csv   # 패러프레이즈 탐지 테스트 데이터
├── sonnets.txt              # 소넷 생성 학습 데이터
├── sonnets_held_out_dev.txt # 소넷 생성 검증 데이터
├── TRUE_sonnets_held_out_dev.txt # 소넷 생성 정답 데이터
├── ids-sst-train.csv        # 감정 분류 SST 학습 데이터
├── ids-sst-dev.csv          # 감정 분류 SST 검증 데이터
├── ids-sst-test-student.csv # 감정 분류 SST 테스트 데이터
├── ids-cfimdb-train.csv     # 감정 분류 CFIMDB 학습 데이터
├── ids-cfimdb-dev.csv       # 감정 분류 CFIMDB 검증 데이터
└── ids-cfimdb-test-student.csv # 감정 분류 CFIMDB 테스트 데이터
```

## 실행 결과 확인 방법

### 1. 학습 과정 모니터링
```python
# 학습 중 출력되는 정보 확인
Epoch 0: train loss :: 0.693, dev acc :: 0.523
Epoch 1: train loss :: 0.645, dev acc :: 0.587
...
```

### 2. 결과 파일 위치
```
predictions/
├── para-dev-output.csv      # 패러프레이즈 탐지 검증 결과
├── para-test-output.csv     # 패러프레이즈 탐지 테스트 결과
├── generated_sonnets.txt    # 생성된 소넷
├── last-linear-layer-sst-dev-out.csv    # SST 감정 분류 검증 결과
├── last-linear-layer-sst-test-out.csv   # SST 감정 분류 테스트 결과
├── last-linear-layer-cfimdb-dev-out.csv # CFIMDB 감정 분류 검증 결과
└── last-linear-layer-cfimdb-test-out.csv # CFIMDB 감정 분류 테스트 결과
```

### 3. 모델 파일 위치
```
├── 10-1e-05-paraphrase.pt   # 패러프레이즈 탐지 모델
├── 10-1e-05-sonnet.pt       # 소넷 생성 모델
├── sst-classifier.pt        # SST 감정 분류 모델
└── cfimdb-classifier.pt     # CFIMDB 감정 분류 모델
```

## 평가 지표 해석

### 1. 패러프레이즈 탐지 (Paraphrase Detection)
- **Accuracy**: 정확도 (0~1, 높을수록 좋음)
- **F1-Score**: 정밀도와 재현율의 조화평균 (0~1, 높을수록 좋음)
- **예상 성능**: Accuracy 0.75~0.85, F1 0.70~0.80

### 2. 소넷 생성 (Sonnet Generation)
- **CHRF Score**: 생성된 텍스트의 품질 평가 (0~1, 높을수록 좋음)
- **예상 성능**: CHRF 0.15~0.25
- **생성 품질**: 문법적 정확성, 의미 일관성, 운율 확인

### 3. 감정 분류 (Sentiment Classification)
- **SST (Stanford Sentiment Treebank)**
  - Accuracy: 0.80~0.90 (5-class 분류)
  - 예상 성능: Accuracy 0.85~0.90
- **CFIMDB (Chinese Film IMDB)**
  - Accuracy: 0.85~0.95 (2-class 분류)
  - 예상 성능: Accuracy 0.90~0.95

## 예상 오류와 해결책

### 1. GPU 메모리 부족 오류
```
RuntimeError: CUDA out of memory
```
**해결책:**
```bash
# batch_size 줄이기
!python paraphrase_detection.py --use_gpu --batch_size 8
!python sonnet_generation.py --use_gpu --batch_size 4
!python classifier.py --use_gpu --batch_size 4
```

### 2. 데이터 파일 없음 오류
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/quora-train.csv'
```
**해결책:**
```python
# 데이터 파일 업로드
from google.colab import files
files.upload()  # data 폴더의 CSV 파일들 업로드
```

### 3. 패키지 설치 오류
```
ModuleNotFoundError: No module named 'transformers'
```
**해결책:**
```bash
!pip install -r requirements.txt
!pip install transformers==4.46.3
```

### 4. 모델 다운로드 오류
```
OSError: We couldn't connect to 'https://huggingface.co/...'
```
**해결책:**
```python
# 인터넷 연결 확인 후 재시도
import requests
requests.get('https://huggingface.co')
```

## GPT-2 성능을 높이기 위한 파인튜닝 적용 방법

### 1. LoRA (Low-Rank Adaptation) 

**구현 위치**: `modules/lora.py`

**적용 방법**:
- Attention 모듈의 Query, Key, Value 선형 변환에 LoRA 적용
- 기존 가중치를 동결하고, 저차원 행렬 A(r×in_features)와 B(out_features×r)만 학습
- 수식: `output = x @ W.T + (x @ A.T) @ B.T * (alpha/r)`

**코드 구현**:
```python
# modules/attention.py에서 적용
self.query = LoRALinear(config.hidden_size, self.all_head_size)
self.key = LoRALinear(config.hidden_size, self.all_head_size)
self.value = LoRALinear(config.hidden_size, self.all_head_size)
```


### 2. Adapter 

**구현 위치**: `modules/adapter.py`

**적용 방법**:
- GPT-2 레이어의 Feed-Forward 네트워크 출력에 Adapter 추가
- Bottleneck 구조: hidden_size → bottleneck(64) → hidden_size
- Residual connection으로 원본 출력과 Adapter 출력을 더함

**코드 구현**:
```python
# modules/gpt2_layer.py에서 적용
self.adapter = Adapter(config.hidden_size, bottleneck=64)
output = self.adapter(output) + output  # Residual connection
```

**구조**:
- Down-projection: `hidden_size → 64`
- ReLU 활성화 함수
- Up-projection: `64 → hidden_size`

### 3. Top-p Sampling and Temperature Scaling 

**구현 위치**: `sonnet_generation.py`의 `generate()` 메소드

**Temperature Scaling**:
```python
logits_last_token = logits_sequence[:, -1, :] / temperature
```
- temperature < 1: 확률 분포를 더 뾰족하게 만들어 일관성 증가
- temperature > 1: 확률 분포를 더 평평하게 만들어 다양성 증가
- 기본값: 1.2 (다양성 증가)

**Top-p (Nucleus) Sampling**:
```python
# 확률을 내림차순으로 정렬
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
# 누적 확률 계산
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
# top-p 임계값 이하의 토큰만 선택
top_p_mask = cumulative_probs <= top_p
# 확률 재정규화
filtered_probs = sorted_probs * top_p_mask
filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)
```
- 기본값: 0.9 (누적 확률 90% 이하의 토큰만 고려)
- 낮은 확률의 토큰들을 제거하여 품질 향상

### 4. 역번역(Back-Translation) 데이터 증강

**구현 위치**: `paraphrase_detection.py`

**적용 방법**:
- 영어 → 프랑스어 → 영어 역번역을 통한 데이터 증강
- MarianMT 모델을 사용하여 자연스러운 문장 변형 생성
- 원본과 동일한 의미를 유지하면서 다양한 표현 방식 학습

**코드 구현**:
```python
# 역번역 모델 로드
tokenizer_pivot = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
model_pivot = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
tokenizer_back = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
model_back = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')

# 역번역 함수
def back_translate(text, tokenizer_pivot, model_pivot, tokenizer_back, model_back, device):
    # 영어 → 프랑스어 번역
    inputs = tokenizer_pivot([text], return_tensors="pt", truncation=True, padding=True, max_length=128)
    translated = model_pivot.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    pivot_text = tokenizer_pivot.decode(translated[0], skip_special_tokens=True)
    
    # 프랑스어 → 영어 역번역
    inputs_back = tokenizer_back([pivot_text], return_tensors="pt", truncation=True, padding=True, max_length=128)
    back_translated = model_back.generate(**inputs_back, max_length=128, num_beams=4, early_stopping=True)
    result = tokenizer_back.decode(back_translated[0], skip_special_tokens=True)
    
    return result if result.lower().strip() != text.lower().strip() else None
```

**학습 시 적용**:
```python
# 데이터 증강 실행
if args.use_augmentation:
    augmented_data = augment_data_with_back_translation(
        para_train_data, tokenizer_pivot, model_pivot,
        tokenizer_back, model_back, device,
        augment_ratio=args.augment_ratio,  # 기본값: 0.3 (30%)
        max_augment=args.max_augment_samples  # 기본값: 500
    )
    para_train_data.extend(augmented_data)
```
