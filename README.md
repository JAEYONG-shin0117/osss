# 자연어처리 2025-1 지정주제 기말 프로젝트: GPT-2 구축

팀원 : 2022113591 신재용 2020112036 김상현 2022113596 오유나

1. GPT-2 기본 모델구현 테스트

#### 다음 모듈들을 실행하여 구현을 테스트한다.

* `optimizer_test.py`: `optimizer.py` 구현을 테스트.
* `sanity_check.py`: GPT 모델 구현을 테스트.
* `classifier.py`: 모델을 사용한 감정 분류 수행.
* `paraphrase_detection.py`: 패러프레이즈 탐지 수행.
* `sonnet_generation.py`: 소네트 생성 수행.

**주목**: 사용하는 GPU 사양에 따라 batch_size 같은 하이퍼파라미터를 조정하여 성능을 최적화하고 메모리 부족 오류를 방지해야 한다.
2. GPT-2 성능을 높이기 위한 파인튜닝 방법

* LoRA & Adapter
* 데이터증강(EDA)
* Beam search


## 환경 설정
**주목**: .yml 파일의 버전을 변경하지 말것.


#### 파이썬 설치
* anaconda3 를 설치한다.

#### 환경 및 패키지 설치

* conda env create -f env.yml
* conda activate nlp_final  

