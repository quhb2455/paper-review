# EfficientNet

### Paper

[https://arxiv.org/pdf/1905.11946.pdf](https://arxiv.org/pdf/1905.11946.pdf)

### Date

21-04-06 ~ 21-04-08

---

# 1. **Introduction**

- CNN은 resource에 여유가 있다면 scale up을 하는데 보통 Width, Depth, Resolution(이하 W, D, R)을 키움.

- network에서 모든 차원(W/D/R)의 균형이 매우 중요하기 때문에 원칙적으로 늘려줘야하지만 이전 연구들은 수작업으로 하나하나 조절해주기 때문에 원칙적이지 않고 효율성도 없음.

- 이 논문에선 원칙적이지 않은 수작업으로 scaling 하지 않고 원칙적이고 효율적인 scaling으로 높은 정확도와 적은 FLOPs를 가지는 method와 network를 소개함.

---

# 2. Related Work

## 1) ConvNet Accuracy : How to scale up a model?

- 최근 SOTA를 달성한 Gpipe를 보면 557M개의 parameter를 가지는데 이런 model은 특별한 training algorithm와 고성능 GPU가 필요함.

    Gpipe의 경우를 보면 이미 H/W는 한계에 달했고 H/W적 한계를 넘기 위해선 scaling을 보다 효율적으로 해야함.

## 2) ConvNet Efficiency : Scaling with principle

- model Compression은 model의 연산 효율을 높이기 위한 방법 중 하나이다 그리고  model size를 줄여 mobile 환경에서 사용하기 위한 보편적인 방법이고 보통 수작업으로 진행되며 W, D, Kernel type and size등을 조절하여 높은 정확도를 가지는 model들이 많이 만들어짐.

    ex) MobileNet, SqueezeNet, ShuffleNet 등등..

    원칙적이지 않은 수작업 방식으로는 더 큰 model에 어떻게 적용해야 할 지 불분명하기 때문에 큰 model을 위한 효율적이고 원칙적인 설계에 초점을 맞춤.

## 3) Model Scaling : Scaling W/D/R

- ResNet과  WideResNet을 보면 Depth와 Widht가 ConvNet 표현력에 큰 영향을 미치는 것을 알 수 있음.

    Width와 Depth의 영향력은 이미 증명되었고, 이 논문은 Resolution까지 더해서 3개의 dimension을 원칙적이고 효율적으로 scaling 하는 방법을 연구함.

---

# 3. Compound Model Scaling

## 1) Problem Formulation

- i번째 ConvNet layer는 $Y_i=F_i (X_i )$ 로 표현함.

    —> $Y_i$ 는 output, $F_i$ 는 연산, $X_i$ 는 input임.

- ConvNet은 이전 layer의 output이 다음 layer의 input이 되므로 아래 수식으로 표현 가능함.

    ![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-1.png](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-1.png)

- 다수의 layer는 stage로 분할되기도 하는데 하나의 stage에 속한 layer는 같은 구조를 가짐.

    ![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-2.png](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-2.png)

    —> $L_i$는 해당 구조를 몇 번 반복할 것인가를 나타냄.

    —> <$H_i, W_i, C_i$>는 input X의 shape.

    —> 이런 구조는 layer가 내려갈 수록 공간차원(W,H)는 줄고 Channel은 커짐

    ex) input=(224,224,3) ⇒ output=(7,7,512)

- 이전의 연구에선 $L_i$$,$$H_i, W_i, C_i$ 를 조절하여 구조를 만들었는데 큰 model에서는 알맞는 $L_i$$,$$H_i, W_i, C_i$를 찾기가 어려움.

    ![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-3.jpg](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-3.jpg)

    —> $w, d, r$ 은 W, D, R의 scaling coefficient
    —> $\hat{F_i}, \hat{L_i}, \hat{H_i}, \hat{W_i}, \hat{C_i}$ 는 predefined parameter([Table 1]())

    모든 층을 일정한 비율로 scaling 되도록 제한을 두어서, 주어진 모든 resource 제약에서 정확도를 최대화 하는게 목적임.

## 2) Scaling Dimensions

- Depth($d$)

    ![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-4.png](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-4.png)

    - ConvNet에서 가장 많이 쓰는 Scaling 방법임.
    - 깊은 ConvNet은 풍부하고 더 복잡한 feature를 잘 찾을 수 있음.
    - 층이 너무 깊으면 Vanishing Gradient 문제로 학습이 힘듬.
    - 깊어 질 수록 정확도 증가폭 감소함.

        ex) resnet-101과 1001의 정확도는 비슷함.

- Widht($w$)

    ![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-5.png](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-5.png)

    - 작은 network size를 위해 자주 사용함.
    - 넓은 ConvNet은 세밀한 feature를 잘 찾고 학습이 쉬움.
    - 너무 넓고 얇은 Network는 높은 level의 feature를 못 찾음.
    - 넓어 질 수록 정확도의 증가폭이 작아짐.

- Resolution($r$)

    ![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-6.png](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-6.png)

    - high resolution image는 detection에서 자주 사용함.
    - resolution이 클 수록 세밀한 패턴을 찾기 쉬움.
    - resolution이 커질수록 연산량이 급격히 증가함.

    위 분석을 통해 $w, d, r$ 중 어떤 것을 키워도 정확도가 상승 하지만 너무 키우게 되면 정확도보다 연산량이 더 많이 증가함.

## 3) Compound Scaling

![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-7.png](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-7.png)

- $w$만 변경했을 때에는 빠르게 saturation 되지만 $d, r$을 함께 변경하면 $w$만 변경했을 때보다 정확도가 높아짐.
- 각 dimension($w,d,r$)은 독립적이지 않기 때문에 dimension간에 균형을 잘 맞춰야함.

    —> $r$을 키우게 되면 큰 image에 담겨있는 정보를 layer에 담아야 하기 때문에 layer가 증가해야함. (Depth 증가)

    —> $r$을 키우게 되면 큰 image에 있는 세밀한 패턴을 읽기 위해서 많은 channel이 필요함. (Width 증가)

![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-13.jpg](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-13.jpg)

- Compound Scaling과 1-Dimenstion scaling을 MobileNet과 ResNet에 적용 했을 때의 차이를 보여줌.
- FLOPs의 차이은 거의 없지만 정확도가 조금이나마 증가하는 것을 볼 수 있음.

위 분석을 통해 정확도와 연산량 효율을 위해 균형 맞춰 $w, d, r$을 scaling 해줘야함.

---

# 4. Compound Scaling Method

- compound coefficient $\phi$를 이용하여 원칙적이고 체계적으로 scaling함.

    ![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-8.png](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-8.png)

    - $\alpha, \beta, \gamma$는 grid search로 찾은 정수
    - $\phi$ 는 사용자가 정하는 계수로 보유한 resource 만큼 정해주면됨.

- ConvNet은 Scaling 할 때 FLOPs가 $(\alpha \cdot \beta^2 \cdot \gamma^2)^\phi$ 로 증가함.
- Compound Scaling Method의 경우에는 $\phi$에 대해 $\alpha \cdot  \beta^2 \cdot \gamma^2 \approx 2$ 로 제한하여 전체 FLOPs는 $2^\phi$로 증가함.
- 특정한 device에 중점을 두지 않기 때문에 latency보다 FLOPs 최적화에 집중함.

---

# 5. EfficinetNet Architecture

![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-11.jpg](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-11.jpg)

- EfficientNet의 baseline network인 EfficientNet-B0의 main block은 mobile inverted bottleneck MBConv이고 여기에 Squeeze와 Excitation optimization이 추가됨.

## 2) Scale up step

### STEP 1

- $\phi$ = 1로 고정시키고 이 [2번 식]() 과 [3번 식]()을 기반으로 $\alpha=1.2,  \beta=1.1,  \gamma=1.15$의 값을 grid search를 통해서 찾음.

### SETP 2

- SETP 1에서 찾은 $\alpha, \beta, \gamma$를 고정시키고 [3번 식]()을 기반으로 $\phi$를 변경하면서 B1 ~ B7의 model을 얻음.

큰 model에서 $\alpha, \beta, \gamma$를 바로 찾을 수 있지만 resource가 많이 들기 때문에 STEP 1과 2로 나눔. 
작은 baseline network에서 한번만 search하기 위해 STEP 1을 수행하고, 같은 scaling coefficients를 다른 모든 model에 적용하기 위해 STEP 2를 함.

---

# 6. Compare Performance

![EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-12.jpg](EfficientNet%20142a7d558c2b41d696ee0056cd928022/effi-12.jpg)

B0 ~ B7 model을 비슷한 정확도를 가진 model들과 비교한 표

- parameter와 FLOPs를 보면 EfficientNet이 매우 작은 것을 알 수 있음.
- model size 또한 EfficientNet이 다른 model보다 매우 작은 것을 알 수 있음.

---

# 7. ConClusion

- 체계적이고 원칙적인 Compound Scaling Method를 통해서 높은 정확도와 효율성을 가진 EfficientNet를 개발함.
- Width와 Depth간의 관계성을 기반으로 Width, Depth, Resolution의 관계성을 알아냈고 이 3가지 dimension이 균형을 이룰때 Network의 성능이 좋아짐.
