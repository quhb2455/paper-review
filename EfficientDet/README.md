# EfficientDet

### Paper

[https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)

### Date

21-03-23 ~ 21-03-26

---

# 1. **Introduction**

- 쉽고 빠르게 multi scale의 feature를 합치는 방법을 제시
**BiFPN (Bidirectional Feature Pyramid Network)**

- Width, height, resolution 뿐만 아니라 Backbone, feature network 그리고 box/class prediction network를 동시에 균일하게 scale up하는 method를 제시
**Compound scaling feature fusion**

- model size가 크고 computing resource가 많이 필요한 model은 real world에선 제약이 많다. 왜냐하면 real world에서는 resource의 제약이 있기 때문에 model이 효율적으로 구성되어야 한다.

- Resource의 제약을 없애기 위해 이전에 연구된 model들은 one-stage, anchor-free detector, compress existing model들을 사용했지만, 이런 method들은 효율은 좋아 졌지만 정확도 부분을 포기해야 했다.

---

# 2. **What do they aim to?**

- 높은 정확도와 효율성을 가지고 스케일링이 가능한 구조가 있는가에 대해 초점을 맞추고 이 논문은 여기에 대해 2가지 주제를 던짐.

---

# 3. Challenge

## 1) Efficient multi-scale feature fusion

- multi-scale feature fusion에서는 FPN이 널리 사용되고 있다.
- 이전에 사용했던 muti-scale feature fusion은 다른 resolution을 가진 각기 다른 input들을 그냥 합치기만 했다. 하지만 resolution을 고려하지 않고 다른 input들을 그냥 합치면 이 input들은 resolution이 다르기 때문에 output에 불균등하게 기여한다는 것을 관찰했다.
- Output에 대한 불균등 기여를 막기 위해 Top-down과 Bottom-up을 반복 수행하여 각기 다른 input feature의 중요도를 학습하는 weight를 가진 BiFPN을 제시한다.

## 2) Model scaling

- 이전의 연구들은 큰 backbone과 큰 input image size에 의존하여 정확도를 상승 시켰다. 그러나 이런 방법들로 scale을 키운 feature network와 prediction network들은 정확도와 효율성 부분에서 매우 크리티컬 한 것을 관찰했다.
- EfficinetNet이 보여준 성능을 기반으로 Width, height, resolution, Backbone, feature network 그리고 box/class prediction network들을 동시에 scale을 키우는 Compound scaling method를 제시한다.

---

# 4. Related Work

## 1) One-stage Detector

- 1-stage가 가진 낮은 정확도라는 특성을 최적화된 network 구조로 극복하여 높은 정확도를 가지고 높은 효율성 또한 가질 수 있다는 것을 보여 줄 것이다.

## 2) Multi-scale Feature Representation

- Muti-scale을 처리하고 표현하는 부분이 object detection에서 어려운 부분인데 초기 model들은 feature pyramid로부터 나오는 feature들을 기반으로 바로 predict을 수행하여 multi-scale 문제를 처리하였다.
- 최근에는 NAS-FPN이 높은 performance를 보여줬는데, 이것은 시간이 오래걸리고 결과가 불규칙적이다.
- 이 논문은 직관적이고 원칙적인 방법을 가진 optimize multi-scale feature fusion으로 multi-scale 처리를 보여 줄 것이다.

## 3) Model Scaling

- 보통 높은 정확도를 얻기 위해선 큰 Backbone을 사용하거나 하는데 최근에는 channel size를 키우거나 feature network를 반복해서 높은 정확도를 얻는다.
- 이 논문은 EfficientNet이 보여준 scale-up 방법을 통해서 model scaling을 할 예정이다.

---

# 5. BiFPN

## 1) Cross-Scale Connections

![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_1.jpg](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_1.jpg)

- (a)는 전통적인 FPN 구조로 Top-down 형식을 가진다.

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled.png](EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled.png)

    - 각 층의 input feature resolution은 $1/2 ^ i$ 형태를 가진다. (i는 층의 level)
    - Resize는 resolution match를 위한 up-sampling, down-sampling을 의미한다.

- (b)는 FPN에서 정보가 한 방향으로 흐르는 걸 해결하기 위해 Bottom-up 구조를 추가한 PANet이다.

- (c)는 NAS-FPN으로 scale이 다른 경우에도 connection이 있는 cross-scale connetion을 적용하 고 있지만 block을 통과하는데 시간이 오래걸리고 결과가 불규칙적이다.

- (d)는 이 논문에서 제안하고 있는 BiFPN 구조이다.
    - Feature fusion이 1개 밖에 없는 node는 feature fusion을 통한 feature network에 기여하는 부분이 적기 때문에 삭제한다.
    - 추가적인 연산 비용없이 feature를 합치기 위해 Original input에서 output으로 가는 edge를 추가한다.
    - 각각의 Bi-direction을 1개의 feature network로 보고 더 높은 level의 feature fusion을 위해 반복 수행한다.

## 2) Weighted Feature Fusion

- BiFPN에선 input feature의 중요도를 학습시킨다.
    - Unbounded Fusion

        ![EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled%201.png](EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled%201.png)

        - W는 Scalar (per feature), Vector (per-channel), Tensor (per-pixel)가 될 수 있다
        - 여기선 최소의 compution resource로 다른 것들과 비슷한 정확도를 달성하는 scale을 실험을 통해 찾아냈다.
        - Scalar weight는 unbounded라서 훈련이 불안정하다. 그래서 범위를 제한하기 위해 weight를 normalize한다.

- Softmax-based Fusion

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_2.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_2.png)

    - Softmax를 각 weight에 적용하는데 과도한 softmax 사용은 GPU를 느리게 한다
    - 그래서 여기선 Fast fusion approach를 제안한다.

- Fast normalized Fusion

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_3.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_3.png)

    - Softmax를 각 weight에 적용하는데 과도한 softmax 사용은 GPU를 느리게 한다
    - 그래서 여기선 Fast fusion approach를 제안한다.

- 위에서 설명한 이점 때문에 본 논문에선 scale값(feature) 을 가지고(Fast normalized fusion을 사용한다.

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_4.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_4.png)

    - Fast normalized Fusion을 사용하게 되면 위의 수식이 적용된다.
    - td는 top-down이다.

**⇒ BiFPN = Bi-directional cross-scale connections + fast normalized fusion**

---

# 6. Compound Scaling

- 큰 backbone과 큰 input image size를 사용하는 것은 1개 혹은 제한적인 dimension에만 국한됨
- compound coefficient φ 를 사용하여 위에서 설명한 모든 network 부분을 동시에 scale up 한다.

## 1) Backbone network

- Backbone은 EfficientNet을 사용한다. 이미 imageNet Classification에서 높은 성능을 자랑하는 network이다.

## 2) BiFPN network

- Depth는 작은 정수로 반올림 되기 때문에 선형 증가시킨다.
- Width는 EfficientNet과 같이 빠르게 증가시킨다. 증가 계수는 grid search에 따르면 1.35가 가장 좋다.

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_5.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_5.png)

## 3) Box/Class Prediction network

- Depth는 선형 증가시킨다.
- Width는 BiFPN과 동일하다.

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_6.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_6.png)

## 4) Input Image Resolution

- Input은 BiFPN이 backbone의 level 3 ~ 7 feature을 이용하기 때문에 항상 2^7 = 128 로 나누어져야한다.

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled%202.png](EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled%202.png)

---

# 7. Compare

- **EfficientDet의 각 버전에 따른 Scaling config. 위의 수식들을 다 적용하면 아래 table의 값이 나온다.**

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_8.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_8.png)

- **Backbone과 BiFPN의 성능비교**

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_9.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_9.png)

    - ResNet + FPN에서 backbone을 efficientNet으로 교체, 마지막으로 FPN 교체로 성능을 비교한다.

- **Cross-scale connections 성능 비교**

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_10.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_10.png)

    - 같은 backbone을 사용했을 때 Cross-scale connection에 따른 model의 복잡도와 정확도를 비교한다.

- **Fast normalized Fusion 성능비교**

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_11.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_11.png)

    - Normalized weight의 이점을 유지하면서 sotfmax fusion에 근접한 성능을 내고 속도가 빠르다.

- **Compound scale up 성능 비교**

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/effi_12.png](EfficientDet%202e512e064a894df58e1136d30e77d176/effi_12.png)

    - 1개의 차원(대상)만 scale을 키웠을 때와 복합적으로 scale을 키웠을 때의 성능을 비교한다.

    ---

    # 8. Conclusion

    본 논문은 BiFPN과 Compound scale-up을 통해서 연산량을 크게 증가시키지 않으면서 multi-scale까지 처리하는 model을 설계 했으며 매우 적은 parameter 개수로 정확도 측면에서 SOTA를 달성하였다.

    yolov4와 성능을 비교해보면

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/yolov4.png](EfficientDet%202e512e064a894df58e1136d30e77d176/yolov4.png)

    전체적인 성능을 비교 했을 때 yolov4보단 아래에 있는 모습을 보여주지만 

    ![EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled%203.png](EfficientDet%202e512e064a894df58e1136d30e77d176/Untitled%203.png)

    backbone parameter와 BFLOPs를 보면 굉장한 차이가 나는 것을 알 수 있다.

    GPU resource가 있는 곳에선 yolov4가 더욱 강세를 보이지만 computing resource가 적은 mobile기기 같은 경우에는 EfficientDet이 더 높은 속도를 보인다.
