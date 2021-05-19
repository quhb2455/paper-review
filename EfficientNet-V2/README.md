# EfficientNet-V2

### Paper

[https://arxiv.org/pdf/2104.00298.pdf](https://arxiv.org/pdf/2104.00298.pdf)

### Date

21-04-21 ~ 21-04-27

---

# 1. **Introduction**

- model 과 dataset size가 점점 커지고 있기 때문에 training 효율이 굉장히 중요해짐에 따라 학습효율관련 연구가 증가함.
    1. NFNet : 학습 효율 증가 - BatchNormalization을 삭제
    2. ResNet-RS : 학습 효율 증가 - Scaling Hyperparameter 최적화
    3. LambdaNet & BotNet : 학습 속도 증가 - ConvNet에서 Attention layer 적용
    4. Vision Transformer : 큰 dataset에서 학습 효율 증가 - Transformer block 사용

        ⇒ 위와 같은 방법이 있지만 보통 많은 parameter가 동반됨.

- Fused-MBConv 를 넓은 search space 확보를 위해 적용하고 Training-aware NAS and Scaling을 통해서 빠른 training 속도와 높은 parameter 효율을 가진 구조를 가진 EfficientNet-V2를 소개함.

- image size에 맞춰서 regularization을 조절하여 정확도와 training 속도를 높여주는 Improved Method of Progressive learning를 소개함.

- EfficientNet-V2 와 Progressive learning을 통해서 ImageNet, CIFAR, Cars, Flowers dataset에서 11배 가량 빠른 training 속도를 보여주고 6배 가량 높은 정확도를 가질 수 있게됨.

---

# 2. Related Work

## 1) Training and Parameter Efficiency

- 이전에는 parameter 효율에 집중해서 적은 parameter로 높은 accuracy를 가지는게 목표였는데 최근에는 training 와 inference 속도에 집중하는 연구가 늘고있음.
- 이전에 나온 training 속도 증가와 관련된 방법들은 대부분 연산량이 많음.
- 이 논문에서는 training 속도는 유지하면서 높은 정확도와 효율성을 가지는 것에 초점을 둠.

## 2) Progressive Training

- progressive resizing과 mix&match 이 두 가지 방법이 우리가 연구한 방법과 가장 근접했는데 이 두 가지 방법들은 똑같은 세기의 regularization을 모든 image size에 동일하게 적용하여 정확도의 하락을 불러옴.
- 여기서 제시하는 방법은 image size가 작으면 약한 regularization을 추가하고, image size가 크면 overfitting을 막기 위해서 강한 regularization을 추가하는 방법으로 image size에 맞게 적응적으로 regularization을 조절함.
- 이 논문은 커리큘럼 학습법(curriculum learning)에서 영감을 받아서 data를 쉬운 것부터 어려운 것 순으로 정렬하는데 이 때 임의로 data를 선별하는게 아닌 regularization을 통해서 training 난이도를 조절함.

## 3) Neural Architecture Search (NAS)

- 이전에는 NAS를 연산 효율과 inference 효율 증가를 위해 사용했지만, 이 논문에선 training과 parameter 효율 최적화를 위해 사용.

---

# 3. EfficientNet V2 Architecture Design

## 1) Understanding Training Efficiency

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled.png)

- image size가 작으면 training시에 batch size를 늘릴 수 있기 때문에 속도가 증가함 그리고 정확도도 아주 조금 증가함.

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%201.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%201.png)

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/v2-3.jpg](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/v2-3.jpg)

- DepthWise Conv는 적은 연산량을 가졌지만 현대의 gpu에선 100% 활용 할 수 없음.
- MBConv에서 Depthwise3x3과 Conv1x1을 일반 Conv3x3으로 교체한게 Fused-MBConv인데 이 연산법이 mobile과 gpu에서 활용성이 더 좋음.
- 체계적인 비교를 위해서 MBConv를 조금씩 Fused-MBConv로 교체함.
    - 초반 stage에 적용 : 학습 속도 증가 및 연산량 증가
    - 모든 stage에 적용 : 학습 속도와 정확도 하락 및 연산량 대폭 증가

    ⇒ NAS를 통해서 자동으로 최상의 조합을 찾음.

## 2) Training-aware NAS and Scaling

- **NAS Search**
    - Search space를 줄이기 위해서 Pooling skip과 같은 불필요한 search option을 삭제
    - EfficientDet에서 연구한 내용을 바탕으로 Backbone에서 사용한 Channel size를 그대로 사용
- **EfficientNet-V2 Architecture**

    ![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%202.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%202.png)

    - V1과 V2의 차이점
        1. V2는 MBConv와 Fused-MBConv를 얕은 층에서 광범위하게 사용
        2. V2는 memory access를 줄이기 위해 작은 expansion ratio를 사용
        3. V2는 receptive field 유지를 위해 layer를 더 추가여 3x3 kernel size를 사용
        4. V2는 큰 parameter와 memory access 때문에 마지막 stride-1 stage를 삭제

- **EfficientNet-V2 Scaling**
    - V2-S를 Scaling하기 위해 몇 가지의 최적화를 추가함.
        1. memory 문제와 training 속도 문제를 해결하기 위해 inference size를 480으로 제한
        2. 추가적인 run time없이 network의 능력을 높이기 위해 stage 5, 6에 layer를 더 추가

- **Training Speed Comparison**
    - Progressive Learning 없이 image size를 고정하여 각 model을 training함

        ![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%203.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%203.png)

        - EfficientNet(reprod)는 image size를 30% 가량 줄인 것이고 EfficienNet(baseline)은 원본 image size로 training 한 것임.
        - Training이 적당히 진행 됐을 때 EfficientNet은 여전히 좋은 성능을 보임
        - EfficieNet V2는 training 속도가 엄청 빠름.

---

# 4. Progressive Learning

## 1) Motivation

- 앞서 봤듯이 image size는 training 효율에 큰 부분을 차지함.
- 이전의 많은 연구들이 training 중에 image size를 동적으로 변화시켜줬는데 가끔 이런 image size변화가 정확도 하락의 원인이 되기도 함.
- 이 논문에선 정확도 하락의 원인이 image size 변화가 아닌 불균형한 regularization이라는 가설을 세움.
- 가설 검증을 위해 search space에서 sampling된 model를 각각 다른 image size와 다른 regularization을 줌.

    ![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%204.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%204.png)

    - image size가 작으면 약한 regularization을 주고 크면 강한 regularization을 주니까 정확도가 상승함.

## 2) Progressive Learning with adaptive Regularization

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%205.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%205.png)

- 초반 epoch에서 작은 image로 학습하고 regularization을 약하게 주면 더 쉽고 빠르게 간단한 표현을 학습 할 수 있음.
- 이전 연구에서도 학습 동안에 image size를 변경했었는데 이 논문에선 여기서 image size에 적응적으로 조절되는 regularization을 추가함.

    ![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%206.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%206.png)

    - 이 논문에선 training을 $M$개 stage로 나눔($1 \leq i \leq M$).
    - target image size = $S_e$ (e는 end를 의미하는 듯)
    - target regularization = {$\phi^k _e$}, $k$ = regularization type
    - $S_0$ 와 $\phi^k _0$는 경험적으로 찾은 값.
    - 각 stage에서의 값을 구하기 위해 선형보간(for문 내에 있는 수식)을 사용
    - ConvNet의 weight는 image size에 독립적이기 때문에 더 쉽게 이전 layer의 값을 상속 받을 수 있음.

## 3) Which kind of regularizaion we use

- **Dropout**
    - network level의 regularization
    - 무작위로 channel을 삭제해서 co-adaptation을 감소시킴
    - $\gamma$로 조절
- **RandAugment**
    - Image augmentation
    - $\epsilon$ 로 조절(epsilon)
- **MixUp**
    - 2장의 image와 label이 있을 때, 비율 $\lambda$ 로 image를 섞고 $\lambda$는 학습 동안 조절됨.

    ex) $(x_i, y_i), (x_j, y_j)$   ⇒   ${\overset{-}x_i }$=$\lambda x_i$ + $(1-\lambda)x_i$, ${\overset{-}y_i }$=$\lambda y_i$ + $(1-\lambda)y_i$

---

# 5. Results

## 1) ImageNet ILSVRC2012

- training setting은 efficientNet과 동일함.

    ![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%207.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%207.png)

- Progressive Learning을 위해 300E을 87E씩 4개의 stage로 나눔.
- min = 첫 stage, max = 마지막 stage를 나타냄.
- 모든 model size에서 min값 setting은 동일 하지만 model이 커질 수록 overfitting을 피하기 위해서 max 값도 같이 커짐 (M-model과 L-model의 max 값 차이).

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%208.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%208.png)

- V2-M 이 EfficientNet-B7보다 학습이 11배 빠름.
- V2 model 모든 ResNet 계열 model보다 정확도나 inference 속도 면에서 우세함.
- training 속도는 inference 속도와 관련있기 때문에 inference도 빠름.

## 2) ImageNet21k

- 높은 정확도 영역에선 model size scaling 보다 dataset size scaling이 더 중요함.
    - model size를 증가시킨다고 정확도가 크게 상승하지 않지만 21k로 pretrained model의 경우([Table 7]() 에 21k라고 model 이름 옆에 표시) 큰 폭으로 정확도가 상승된 것을 알 수 있음.

## 3) Comparison to EfficienNet

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%209.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%209.png)

- V1과 V2 둘다 Progressive Learning일 때, 아주 큰 차이를 보임

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%2010.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%2010.png)

- size가 작은 model끼리 비교 했을 때, V2가 2배이상 빠르다는 걸 알 수 있음.

## 4) Progressive Learning for Different Networks

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%2011.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%2011.png)

- baseline이 원본 model보다 성능이 좋은 이유는 EfficientNet의 training setting을 그대로 했기 때문.
- Default image size가 작으면 training 속도 상승이 크지 않음.
- image size가 커지고 model이 복잡해지면 training 속도에서 큰 차이가 있고 정확도도 조금 상승함.

## 5) Importance of Adaptive Regularization

![EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%2012.png](EfficientNet-V2%200ea2e2fd5bf245739d246c3aa94fe4e7/Untitled%2012.png)

- 각 batch마다 Random하게 resize를 하는 것(Random resize)보다 Progressive resize가 Vanilla approach 부분에서 더 성능이 좋았고 거기에 adaptive regularization을 추가하면 더 좋은 성능을 뽑을 수 있음.
- 위 그래프(figure6)을 보면 progressive + adaptive reg가 더 빨리 수렴하는 걸 볼 수 있음.

---

# 6. Conclusion

- training-aware NAS and model Scaling으로 최적화된 V2는 학습동안 image size에 맞춰서 regularization을 조절하는 Progressive Learning으로 학습 속도를 크게 개선했으며, parameter 효율도 좋아졌음.
- V1과 비교하면 6.8배 작고 11배 빠름.
