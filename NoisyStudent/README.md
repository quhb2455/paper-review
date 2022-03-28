# Noisy Student

생성일: 2022년 3월 21일 오전 8:46

### Paper

[https://arxiv.org/pdf/1911.04252.pdf](https://arxiv.org/pdf/1911.04252.pdf)

### Date

22-03-17 ~ 22-03-18

### R**eference**

[https://joungheekim.github.io/2020/12/13/code-review/](https://joungheekim.github.io/2020/12/13/code-review/)

[https://www.tensorflow.org/tutorials/generative/adversarial_fgsm](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)

### 읽기 전 알아야 할 것들

- Stochastic Depth
    - 일정 확률로 Layer를 생략하고 넘어가는 기법
        
        $p_l = 1 - \frac {l}{L}(1-p_L)$
        
        - $p_L$ = 마지막 Layer가 생략될 확률
        - $L$ = 전체 Layer 개수
        - $p_l$ = 현재 Layer가 생략될 확률
        - $l$ = 현재 Layer 번호
    - Dropout과 유사한 개념
- SSL
    - Semi-Supervised Learning 의 약자
- WSL
    - Weakly-Supervised Learning 의 약자
- Distillation
    - Knowledge Distillation을 의미
    - Teacher 모델을 이용하여 Student 모델을 학습하는 방법
- Consistency Training
    - 원래 의미
        - 모델이 데이터의 작은 변화에 민감하지 않게(Robustness) 만드는 학습법
    - **SSL에서의 의미(=본 논문에서의 의미인듯)**
        - Noisy(=augmentation)가 추가된 Unlabeled data와 Noisy가 추가 되지 않은 Unlabeled data가 동일한 Label을 갖도록 학습하는 것
- Consistency Loss
    - Unlabel image와 Label image 간의 유사도를 계산한 것
    - 높으면 비슷하고 낮으면 다름
- Adversarial Attack
    - 이미지에 noise 값을 넣은 이미지
    - 사람의 눈으로는 별차이 없는데 Feature map으로 출력하거나 혹은 Inference를 진행하면 결과가 달라지는 것을 볼 수 있음

# Abstract

- Noisy Student 는 Self-training과 Distillation에서 확장된 개념
- Self-training, Distillation 과의 차이점
    - Noisy 라는 것을 Student 모델에 추가함
    - Student 모델을 Teacher 모델보다 같거나 큰 사이즈의 모델로 사용함
- 학습 순서
    1. Labeled ImageNet을 EfficientNet에 학습시킴
    2. `Labeled data`로 학습된 EfficientNet을 `Teacher`모델로써 사용
    3. `Teacher` 모델로 `Unlabeled data`(300M 장)를 Pseudo Labeling 진행
    4. `Teacher` 모델과 파라미터 사이즈가 동일하거나 큰 모델을 `Student` 모델로써 사용
        - 논문을 다 읽은 결과.. Unlabeled dataset을 모두 담을 수 있을 정도의 parameter size를 가진 모델로 선택하면 됨
        - Unlabeled data가 많지 않다면 작은 모델에서 시작해서 조금씩 키워가도 될 뜻
    5. `Labeled data`와 `Unlabeled data`로 `Student` 모델 학습
        - **이 때, Noisy라고 불리는 3가지 기법을 추가함(핵심)**
            - `Rand Augmentation`
            - `Dropout`
            - `Stochastic Depth`
    6. 학습이 완료된 `Student`모델을 `Teacher` 모델로써 사용
    7. 3번 ~ 6번까지 2회 ~ 4회 정도 반복

# 1. Introduction

- Unlabeled data를 사용하여 SOTA ImageNet 모델의 정확도와 Robustness를 크게 향상시킴
- Unlabeled data의 많은 부분이 Labeled data의 Distribution에 속하지 않음
    - Labeled data의 class category에 속하지 않는 다는 의미
- Semi-Supervised Learning과 Distillation에 포함된 개념인 Self-Training의 버전을 업그레이드 한 것이 Noisy Student Training 임
- Noisy Student Training이 Self-training, Distillation과 다른 2가지
    1. Student 모델을 Teacher 모델보다 크거나 같은 사이즈의 모델을 사용함
    2. Student 모델이 학습 할 때, Noisy를 추가하여 학습함
        - Rand Augmentation
        - Dropout
        - Stochastic Depth

# 2. Noisy Student Training

![Untitled](Noisy%20Stud%201e888/Untitled.png)

![Untitled](Noisy%20Stud%201e888/Untitled%201.png)

- Teacher 모델을 활용하여 Unlabed data를 Pseudo labeling 할 때, **Soft labeling**으로 진행함
    - Soft Labeling
        - [0.2, 0.5, 0.1] ⇒ 이런 식으로 확률 값을 넣어주는 듯 (Continuous Distribution)
        - Unlabeld data는 Labeled data의 Distribution에 포함되지 않기 때문에 Soft로 넣는게 성능에 좋음
    - Hard Labeling
        - [0, 1, 0] ⇒ 이런 식으로 넣어주는 값 (One-Hot Distribution)
- Knowledge Distillation과 다른점
    - Knowledge Distillation은 Noisy를 사용하지 않음
    - Teacher모델 보다 작은(혹은 더 빠른) Student 모델을 사용함
- Noisy Student Training은 충분한 파라미터(=Capacity, 수용력)와 Noisy를 통한 어려운 환경에서 학습을 진행하여 Teacher 모델보다 더 좋은 Student 모델을 생성함

## 1) Noisy Student

- Noisy가 적용된 Student 모델은 Teacher 모델과 동일하게 학습됨
- Noisy의 2가지 종류
    - Input Noise
        - Data augmentation
            - Augmentation을 적용함으로써 Student 모델이 Augmented data를  original data와 동일한 예측을 하게 만듬
                - **Original data와 관련된 더 어려운 이미지도 잘 예측하게 만듬**
    - Model Noise
        - Dropout
        - Stochastic Depth
        
        ⇒ 위 2가지 방법은 Pseudo labeling 할 때, 모델이 앙상블 효과를 내게함
        따라서, **Student 모델은 Pseudo labeling된 data로 학습하기 때문에** **강력한 앙상블 효과를 모방하는 모델이 됨**
        

## 2) Other Techniques

- Noisy Student Training에선 `Data Filtering`과 `Data Balancing`을 사용함
    - Class 간의 data 개수를 맞추기 위해 사용
- Data Filtering
    - Teacher 모델로 Pseudo Labeling 시에 Confidence score가 낮으면 Unlabeled data로 사용하지 않음
        - Example
            - Unlabeled data의 class는 [사자, 원숭이] 를 가짐
            - Labeled data의 class는 [고양이, 자동차]를 가짐
            - Unlabeled data를 Pseudo labeling 할 때, 사자는 고양이와 가깝기 때문에 높은 Confidence Score를 가짐
            - 반면에 원숭이는 어떠한 class에도 속하지 않기 때문에 낮은 Score를 받음
            - 따라서 낮은 Score를 가진 Data는 Labeled data의 Distribution에 속하는 data가 아니라고 판단(=Out of Domain)하여 지울 수 있음
- Data Balancing
    - Class의 data 수가 적으면 data를 복사하여 사용

## 3) Comparisons With Existing SSL Methods

- SSL은 Consistency Training 과 Pseudo Label를 기초로한 개념임
- SSL과 Noisy Student Training과의 차이점
    - SSL은 Teacher 모델을 따로 분리하지 않고 학습 중에 Pseudo label을 만듬
        - 학습 초기에 모델이 낮은 정확도와 높은 Entropy를 가짐
        - 이 후에, Consistency Training을 통해 모델이 높은 Entropy를 가지는 쪽으로 정규화되어 좋은 성능을 가질 수 없음
    - 위 문제의 해결법
        - Low Confidence Score를 걸러내던가 Consistency Loss를 높혀줘야함
        
        ⇒ 이 해결법은 대규모 학습에선 학습을 더 어렵게 함
        

# 3. Experiments

## 1) Labeled dataset

- ImageNet 2012 ILSVRC 사용

## 2) Unlabeled dataset

- JFT-300M 사용
- Image Filtering 과 Balancing 방법
    - EfficientNet-B0를 ImageNet으로 학습 후 JFT-300M에 대해 confidence score를 출력
    - Confidence Score가 0.3 이상만 사용함
    - Threshold를 통과한 data는 각 class 당 130K 정도
    - 130K가 안되는 class에 대해서는 Duplicate 진행
    - Filtering과 Balancing 이후에 남은 130M 정도의 사진 중 Duplicate 되지 않은 Unique 사진은 81M 정도임

## 3) Architecture

- Baseline 으로 EfficientNet 사용
- B7 → L2 까지 Scale을 키우면서 진행
    - 단, L2의 Input resolution은 낮게 만듬
        
        Why? ⇒ 이미 충분한 Parameter가 있어서 크게 만들 필요가 없었다고 함
        

## 4) Training Details

- Labeled data를 학습 할 때에는 Batch size에 대해서 결과 차이가 없었기 때문에 Memory에 올릴 수 있는 최대 크기로 사용
- Student 모델의 사이즈가 L2보다 크면 300 epoch을 주고 작으면 700 epoch으로 학습함
- Learning rate = 0.128로 시작, Decay를 2.4 Epoch 마다 0.97 씩 진행
- Unlabeled Data의 batch size느 Labeled data의 batch size에 14배 크기로 진행
- train-test 간의 해상도 불일치 문제를 해결하기 위해서
    1. 350E 까지는 작은 Resolution 그대로 학습 진행
    2. 학습이 끝난 후 model 의 resolution을 크게 바꾸고 Unaugmented labeled image에 대해 1.5 E 만 FineTuning 실시
        - 이 때, 초기 Layer는 Freeze 해줌

## 5) Noisy

- Stochastic Depth는 Final layer에 0.8을 주고 이 외의 Layer는 Linear Decay를 따르게 함
- Dropout rate = 0.5
- RandAugmentation에서 2개의 Augmentation을 뽑아서 사용했고, Magnitude는 27로 줌

## 6) Iterative Training

- 전체 과정을 3번 반복하는게 베스트
- 전체 과정이 1번 반복되면 unlabeled data의 batch size를 28배로 올려줌(원래는 14배)

# 4. Results

## 1) **Noisy Student Training for EfficientNet-B0 ~ B7 Without Iterative Training**

![Untitled](Noisy%20Stud%201e888/Untitled%202.png)

- Iterative 없이 Student 모델과 Teacher 모델을 동일한 모델로 사용 했을 때에도 Noisy Student Training을 통해서 모든 모델의 정확도 향상이 있음

## 2) Robustness Results On ImageNet-A, C, P

![Untitled](Noisy%20Stud%201e888/Untitled%203.png)

![Untitled](Noisy%20Stud%201e888/Untitled%204.png)

![Untitled](Noisy%20Stud%201e888/Untitled%205.png)

![Untitled](Noisy%20Stud%201e888/Untitled%206.png)

- Robustness Benchmark에 이용되는 Dataset
    - ImageNet-P
        - test image에 Rotation과 Image Scale up이 적용되어 있음
    - ImageNet-C
        - test image에 강한 Noise가 끼여 있음
    - ImageNet-A
        - test image가 어려움
- ImageNet- A, C, P를 통해서 어려운 data, noise가 있는 data, scale이 다른 data에 대해서 잘 작동하는 것을 보여주며 Robustness가 좋다는 것을 말하고 있음
- Adversarial Attack에 대해서도 잘 작동하는 것을 통해 역시 Robustness가 좋다는 것을 알 수 있음
    
    ![입실론은 image 왜곡의 강도를 나타냄](Noisy%20Stud%201e888/Untitled%207.png)
    
    입실론은 image 왜곡의 강도를 나타냄
    

# 5. Ablation Study

## 1) The Importance of Noise in Self-training

![Untitled](Noisy%20Stud%201e888/Untitled%208.png)

- Noise를 제거하면 할 수록 정확도가 떨어지는 것을 볼 수 있음
    - Student 모델에 대해서 RangAug만 뺏을 때보다 3가지 Noisy를 모두 뺏을 때 정확도가 더 낮음
- Teacher 모델에 대해서 Noisy를 주면 정확도가 떨어짐

## 2) A Study of Iterative Training

![Untitled](Noisy%20Stud%201e888/Untitled%209.png)

- Hyperparameter를 그대로 유지하더라도 1번의 반복을 통해서 정확도가 0.5% 상승하는 것을 볼 수 있음

# 6. Study Summarization

1. **Teacher 모델이 클 수록 좋음**
2. **Unlabeled data가 클 수록 좋음**
3. **Pseudo labeling시 Soft labeling이 Hard labeling 보다 성능이 좋음**
4. **Student 모델이 클 수록 더 강력한 모델을 만들 수 있음(1번과 맥락이 비슷)**
5. **작은 모델에 대해서는 Data Balancing이 중요함**
6. **Unlabeled와 Labeled를 함께 훈련하는게 Unlabeled로 pretraining 하고 Labeled로 finetuning하는것 보다 성능이 좋음**
7. **Unlabeled data에 대해서 더 큰 Batch size를 사용하는게 더 높은 정확도를 얻을 수 있음**
8. **Student 모델을 처음부터 훈련시키는게 Teacher로 Student를 훈련시키는 것보다 나을 때가 있음**
9. **Teacher 모델로 Student를 초기화 시킬 때에는 높은 Epoch이 필요함**

# 7. Related Work

## 1) Self-Training

- Noisy Student Training과 Self-Training의 차이점
    - Noisy을 Student에 적용하는 것
        - Self-Training에선 Noisy를 사용하지 않는게 Default 이며 명확하게 Noisy에 대해서 정의되어 있지 않음
- Self-Training의 문제점
    - SOTA에 비해 너무 낮은 정확도
    - Robustness에 대한 향상이 없음
    - Unlabeled data로 먼저 학습을 하고, Labeled data로 Finetuning을 함
- Data Distillation
    - Teacher를 강화하기 위해 이미지에 Augmentation을 적용하는 방법
    - Student를 약하게 만들기 때문에 Noisy Student Training과 완전 반대임
- Co-training
    - Labeled data의 Feature를 결합할 수 없는 2개의 부분으로 나누고 각각 모델을 학습시키는 방법
    - 2개의 모델이 Unlabeled data에 대해서 항상 동일한 결과를 보여주지 않음

## 2) Semi-Supervised Learning

- SSL은 Consistency Training을 기반으로 함
- Consistency Regularization을 사용함
    - 모델 훈련 중에 Pseudo labeling을 실시하여 학습 초기에 낮은 정확도와 높은 Entropy를 가지게 함
- 대부분의 SSL 기반의 모델은 완전히 수렴된(학습이 완료된) 모델로 Pseudo labeling을 진행하는게 아니고 학습 중에 Pseudo labeling을 진행

## 3) Knowledge Distillation

- Soft target을 사용한다는 점이 Noisy Student와 비슷함
- Student 모델을 작게 만들어서 모델 압축에 많이 사용하는 기술임
    - Noisy Student는 Student를 크게 만드는 기법이기 때문에 완전 반대되는 기법임

## 4) Robustness

- Unlabeled data를 통해서 Robustness하고 높은 정확도를 가지고 있으며, Adversarial Robustness도 가짐을 알 수 있음
- Noisy Student Training은 Robustness를 직접적으로 최적화하지 않지만 강한 Robustness를 보여줌

# 8. Conclusion

- WSL을 기반으로 하는 이전 연구는 SOTA를 달성하기 위해 많은 양의 Weakly Unlabeled data가 필요했음
- Unlabeled data가 SOTA 모델의 정확도와 Robustness를 향상시킴을 보여줌
- Self-Training과 거의 유사하지만 Nosiy를 Student에 추가한 점이 다름
    - 따라서 이름을 Noisy Student라고 부름
- **Noisy Student Training의 적용을 통해서 기존의 SOTA 모델 정확도를 2% 가량 올려줌**