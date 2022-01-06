# CutMix

### Paper

[https://arxiv.org/pdf/1905.04899.pdf](https://arxiv.org/pdf/1905.04899.pdf)

### Date

2021-11-17 ~ 2021-11-19

### R**eference**

- [https://jjerry-k.github.io/blog/cutmix/](https://jjerry-k.github.io/blog/cutmix/)
- [https://hongl.tistory.com/223](https://hongl.tistory.com/223)

### Implementation

🔎[Code](https://www.notion.so/CutMix-8f38580fb8504b5abc0eabfdf48b90c2)

### 읽기 전 알아야 할 것들

- Patch : 학습 중에 사용할 1개의 이미지에서 잘라낸 부분으로 또 다른 학습 이미지에 붙이게 됨.

# Abstract

- 2019년에 Naver Clova 에서 발표한 논문
- 현재 사용되는 regional dropout method들은 정보를 담고 있는 픽셀을 검정 픽셀로 오버랩 하여 지운다거나 랜덤 노이즈 값을 오버랩 함.
- 픽셀을 강제적으로 지우는 것은 이미지가 가지고 있는 정보를 손실 시키는 행위이기 때문에 학습 간에 효율성이 떨어짐.
- Cutmix는 학습 중에 사용되는 이미지의 랜덤 부분을 잘라서 다른 이미지에 붙이는 방식을 사용함.
- 2개의 이미지를 섞을 때 잘라내는 크기에 비례하게 GT label 값을 수정해줌.
- 픽셀을 강제로 없애거나 하지 않기 때문에 정보의 손실이 덜하고 regional dropout의 정규화 효과도 유지하기 때문에 학습의 효율성을 유지할 수 있음.
- Cutmix를 사용하게 되면 일반화 성능이 좋아져서 모델의 robustness가 상승함.

# 1. Introduction

- CNN이 입력 이미지의 작은 부분에 너무 집중하는 것을 방지하기 위해 random feature removal regularization이 제안 됐음.
- feature removal regularization은 모델이 object의 차별적인 부분뿐만 아니라 전체적인 부분에 대해서도 잘 일반화 되고 지역화되게 함.
- 일반적으로 CNN 모델들은 학습 데이터가 부족하기 때문에 학습 이미지의 정보를 손실시키는 행위는 학습에 치명적임 그래서 최대한 학습 이미지에 포함된 정보들이 손실이 되지 않는 범위에서 regularization을 진행해야함.
- Cutmix는 학습 간에 정보가 없는 픽셀은 없다라는 특성을 이용하여 학습을 효율적으로 만들고 동시에 regularization의 이점을 유지함.
- 이미지에 Patch를 추가하게 되면 partial view로 object를 식별하여 모델의 지역화 능력이 더 향상됨.
- Cutmix는 학습과 Inference 간에 Computing resource가 거의 들지 않음.
- Mixup 기법을 사용하여 생성된 sample은 부자연스러움. Cutmix는 이것을 극복하기 위해 다른 학습 이미지에서 patch를 가져와 현재 이미지에 붙여 넣는 방법을 사용함.
- Mixup과 Cutout은 Classification 능력은 향상 시켰을지 몰라도 Detection 능력은 떨어트림.
- Cutmix가 성능 향상을 시킨 것
    - Baseline classifier의 성능
    - Weakly-Supervised Object Localization(WSOL)의 성능
    - Detection의 전반적인 성능
    - 모델의 Robustness
    - Over-confidence issue

# 2. Related Works

## Regional Dropout

- Cutmix는 다른 method들과 비슷하지만 삭제된 부분을 다른 학습 이미지로부터 Crop한 patch로 다시 채워 넣는 것이 가장 다른 부분임.

## Synthesizing Training  Data

- Stylizing ImageNet으로 만들어진 새로운 학습 데이터들은 모델이 데이터의 shape 보다는 texture에 집중하도록 해서 더 나은 classification과 detection 성능을 끌어냄.
⇒ CutMix도 학습 데이터를 생성해내지만 Stylization과 다르게 Computing Resource를 더 적게 먹음.
- Detection에서 Object Insertion method는 Object를 배경에 합성하여 새로운 데이터를 만듬.
⇒ Object Insertion method는 1개의 Object에만 집중하지만 CutMix는 다수의 Object에 집중 할 수 있음.

## Mixup

- Mixup은 CutMix와 꽤나 비슷하지만 만들어진 데이터가 애매모호하고 부자연스러운 단점이 있음.
- Mixup에서 파생되어 나온 다른 연구에서는 localization 능력과  Trasfer-Learning 성능에 대한 심층분석이 부족함.

## Tricks for Training Deep Networks

- 최근에는 Classification의 성능을 개선하기 위해 CNN의 내부 feature에 noise 값을 넣는다던가 Architecture에 다른 path를 추가하는 method들이 있음.
⇒ CutMix는 위의 2가지 method들을 보완할 수 있음.

# 3. CutMix

## Algorithm

- CutMix의 목표는 2가지 데이터$(x_A, y_A), (x_B, y_B)$ 를 섞어서 새로운 데이터$(\tilde{x}, \tilde{y})$ 를 만든 것임.
- 새로 생성된 데이터$(\tilde{x}, \tilde{y})$는 기존에 사용하던 loss함수를 그대로 이용함.
- 2가지 데이터를 섞는 연산식
    
    $$
    \tilde{x} = M\odot x_A + (1-M)\odot x_B \\
    \tilde{y} = \lambda{y}_A + (1-\lambda){y_B}
    $$
    
    - $M \in {\{0,1}\}^{W\times H}$ 은 2개의 이미지에서 Pixel을 없애거나 다시 채울때 사용하는 Binary Mask임.
    - $\lambda$는 2개의 이미지를 합성할 때 사용하는 비율이고 Beta Distribution인 $Beta(\alpha, \alpha)$에서 sampling 됨.
    ⇒ 논문에선 $\alpha$를 1로 두고, $\lambda$는 Uniform Distribution(0,1)에서 sampling됨.
    - 이미지에서 BBox의 크기만큼 패치를 뜯어냄.
    - M의 크기는 이미지의 크기와 동일함
- BBox의 크기 및 좌표를 샘플링 하는 식
    
    $r_x \sim \mathsf{Unif}(0,W), \qquad r_w = W\sqrt{1-\lambda} \\
    r_y \sim \mathsf{Unif}(0,H), \qquad r_h = H\sqrt{1-\lambda}$
    
    ```python
    import numpy as np
    lam = np.random.beta(1.0 , 1.0)
    
    # 사각형 중심 좌표
    rx = np.random.randint(W)
    ry = np.random.randint(H)
    
    # 사각형 가로세로 길이
    rw = np.int(W * np.sqrt(1 - lam))
    rh = np.int(H * np.sqrt(1 - lam))
    ```
    

## Discussion

- **What does model learn with Cutmix?**
    
    ![Untitled](CutMix%20a9fc654c7eef423dac7d48868b79c092/Untitled.png)
    
    - 훈련 동안 효율성을 높이기 위해 1개의 이미지에서 2개의 객체를 인식함
    - Cutout은 사용하지 않는 픽셀들 때문에 비효율적임
    - Mixup은 이미지가 자연스럽지 않고 겹쳐져 있기 때문에 모델이 어떤 부분에 집중해야 되는지 헷갈려함
    - Cutmix는 1개의 이미지에서 2개의 객체에 잘 집중하고 있음
- **Analysis on validation error**
    
    ![Untitled](CutMix%20a9fc654c7eef423dac7d48868b79c092/Untitled%201.png)
    
    - Cutmix 를 추가하게되면 overfitting이 일어 나지않고 꾸준히 error 값이 감소함
    - 새로 생성된 sample들 덕분에 Overfitting이 안남
    

# 4. Experiments

## Ablation Study

![Untitled](CutMix%20a9fc654c7eef423dac7d48868b79c092/Untitled%202.png)

![Untitled](CutMix%20a9fc654c7eef423dac7d48868b79c092/Untitled%203.png)

- Lambda 값을 구하는 $\alpha$는 1일 때 가장 좋음
- Cutmix를 Input Image에 적용했을 때가 가장 성능이 좋음
- 여러가지 CutMix 방법들
    - Center Gaussian CutMix
        - Image Center에서 정규분포 대신에 Gaussian 분포를 따라서 Image를 Crop 하는 방법
    - Fiexed-size CutMix
        - Crop하는 부분의 크기를 Fix 하여 사용하는 방법
        - $r_w\times r_h = 16\times 16$ 그리고 $\lambda = 0.75$
    - One-hot CutMix
        - 패치 비율에 따라 Portion Label(기존의 Cutmix label 방식)대신에 One-hot Label을 사용
    - **Scheduled CutMix**
        - 학습 중에 Cutmix가 적용될 확률을 0부터 1까지 선형적으로 증가 시키는 것
        - 이 방법이 가장 효과가 좋다고 함
    - Complete-label CutMix
        - $\tilde y$ 를 구할 때 $\lambda$를 쓰지 않고 $\tilde y = 0.5y_A + 0.5y_B$ 로 계산하는 것

## Transferring to Pascal VOC Object Detection

- Cutout과 Mixup를 각각 이용하여 Object Detection 모델을 pretrain 할 때 Vanila-pretrained 모델 보다 성능이 안 좋음
- CutMix는 Localization 부분에서 강력한 성능을 보여줌으로써 Detection 성능을 향상시킴

## Weakly Supervised Object Localization(WSOL)

- WSOL은 Image의 Label만을 이용하여 Object를 지역화하도록 분류기를 훈련하는게 목표임
- Object를 지역화하기 위해선 CNN이 Object를 최대한 넓게 보게하고 작은 부분에 집중하지 않도록 해야함
- WSOL 성능 향상을 위해서 Spatially Distributed Representation(공간적으로 분산된 표현)을 학습하는 것이 중요함

⇒ CutMix는 모델이 더 광범위한 단서(분류를 위한)에 집중하도록 함 이런 부분이 WSOL의 성능을 향상시켜 줄 수 있다고 함

## Robustness and Uncertainty

- Deep Model은 인식할 수 없을 정도로 작은 변화에 쉽게 속아 넘어가는데 이런 현상을 Adversarial Attack 이라함
- Robustness 와 Uncertainty를 증가시킬 수 있는 간단한 방법은 Input Image를 본 적 없는 새로운 sample로 Augmentation 해주는 것임

### **Robustness**

![Untitled](CutMix%20a9fc654c7eef423dac7d48868b79c092/Untitled%204.png)

![Untitled](CutMix%20a9fc654c7eef423dac7d48868b79c092/Untitled%205.png)

- Adversarial sample, Occluded sample, in-between class sample 로 CutMix로 학습한 모델의 Robustness를 검증함
- 위 사진을 보면 Adversarial Atteck에 대해서 다른 method에 비해 높은 정확도를 보여주고 있음 이 부분이 Robustness을 증명 할 수 있는 부분
- Center Occlusion 과 Boundry Occlusion을 적용할 때 Cutout과 Cutmix는 Robustness에 대해서 눈에 띄는 향상을 보여주지만 Mixup은 약간의 상승만 보여줌
- In-between sample에 대해서 Cutmix와 Mixup은 성능향상을 보여줬지만 Cutout은 무시할 수준의 차이를 보여줌 그리고 Cutmix는 본 적없는 Mixup된 데이터에 대한 Robustness 향상을 보여줌

### Uncertainty

![Untitled](CutMix%20a9fc654c7eef423dac7d48868b79c092/Untitled%206.png)

- Out-of-Distribution(OOD) Detector
    - 데이터가 분포 내에 있는지 혹은 분포 외에 있는지 Score Threshold를 통해서 판별함
    - In-Distribution 데이터로 모델을 훈련 시키고 test로는 in-Distribution과 Out-Distribution 데이터셋 모두를 이용하여 Softmax 함수 결과가 가장 높은 값을 Threshold에 따라 in / out을 구분
    - out-distribution 데이터의 경우 모델이 예측 자체를 하지 않아야 되니 클래스별로 동일한(Uniform)확률이 도출되어야 함
- DNN은 분류 예측시 Softmax 함수를 사용하기 때문에 Over Confidence경향이 있음
    - Accuracy보다 Confidence(모델의 Output으로 나온 score)가 높은 것을 말함
    - OverConfidence를 테스트 하기위해 OOD를 사용
- Cutout과 Mixup은 OverConfidence를 강화시켜서 성능이 안 좋아짐
- CutMix는 OOD 데이터에 대해 성능이 제일 좋음

# 5. Conclusion

- CutMix는 구현이 쉽고 학습에 적용할 때 컴퓨팅 리소스도 거의 먹지 않음
- Localization 부분 SOTA인 WSOL과 비교할 수 있을 정도로 Localization에 대한 성능을 많이 끌어올려줌
- CutMix를 통해서 이미지 분류의 Robustness와 Uncertainty를 성능향상이 가능 하다는 것을 보여줌