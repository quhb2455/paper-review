# Batch Normalization

생성일: 2022년 2월 9일 오후 4:43

### Paper

[https://arxiv.org/pdf/1502.03167.pdf](https://arxiv.org/pdf/1502.03167.pdf)

### Date

22-02-08 ~ 22-02-11

### R**eference**

- [https://deep-learning-study.tistory.com/421](https://deep-learning-study.tistory.com/421)
- [https://goodjian.tistory.com/80](https://goodjian.tistory.com/80)
- [https://cvml.tistory.com/5](https://cvml.tistory.com/5)
- [https://blog.naver.com/PostView.naver?blogId=laonple&logNo=220808903260&parentCategoryNo=&categoryNo=22&viewDate=&isShowPopularPosts=false&from=postView](https://blog.naver.com/PostView.naver?blogId=laonple&logNo=220808903260&parentCategoryNo=&categoryNo=22&viewDate=&isShowPopularPosts=false&from=postView)

### 읽기 전 알아야 할 것들

- Activation
    - Activation Function을 의미하는게 아님
    - Feature를 구성하고 있는 1개의 값
        - 해당 논문에선 Scalar Feature를 위주로 설명
    - Scalar Feature 란?
        
        ![d = Dimension](Batch%20Norm%2010e8f/Untitled.png)
        
        d = Dimension
        
- Saturating non-linearity
    - 입력 X가 무한대로 갈 때, 함수값이 범위 안에서만 움직이는 것
    - sigmoid function, Tanh function
- Non-saturating non-linearity
    - 입력 X가 무한대로 갈 때, 함수값이 무한대로 가는 것
    - ReLU
- Whitening
    - Input의 Feature들을 Uncorrelated 하게 만들어주고, 각각의 Mean(평균)을 0, Variance(분산)을 1로 만들어주는 작업
- Covariance Matrix
    - 구조 내의 상관관계를 말해주는 Matrix
    
    ```python
    # X의 Size가 (N, D) 일 때,
    x -= np.mean(x, axis=0)
    cov = np.dot(x.T,x) / x.shape[0]
    # cov 행렬의 (i, j)번 째 원소는 두 차원(i, j)의 상관관계 정도
    ```
    
- Uncorrelated
    - 변수 간에 상관관계가 없음을 뜻함.
    - 참고 : [https://charmed-creek-53c.notion.site/Uncorrelated-c8965fbabd4a4a568ce68020dac2117a](https://www.notion.so/Uncorrelated-c8965fbabd4a4a568ce68020dac2117a)

# Abstract

- DNN의 학습이 복잡한 이유는 학습 동안 이전 층의 Parameter가 계속 변해서 다음 층의 Input값도 계속 바뀌기 때문임
    - 위 문제는 Saturating non-linearity을 사용하는 모델의 학습을 더욱 어렵게 만듬
    - 위 문제를 해결하기 위해선 낮은 Learning rate(이하 LR)를 사용하고 Parameter 초기화에 신경써야함
    - 이 문제를 `Internal Covariate shift`(내부 공변량 변화)라고 정의함.
- Internal Covariate Shift는 layer의 입력을 Normalization 함으로써 해결할 수 있음.
- 이 논문에선 Internal Covariate Shift를 없애기 위해 `Batch Normalization`(이하 BN)이라는 메소드를 발표함
- BN은 Normalization을 모델의 한 부분으로 만들어서 각 Batch 마다 Norm을 수행함

# 1. Introduction

- Stocastic Gradient Decent(SGD)란
    - 모델의 $\Theta$를 Optimize(최적화)하여 Loss값을 줄임
    - 간단하고 효과적이지만 높은 LR을 사용할 수 없고, Hyperparameter의 초기화를 신경써야함
    - SGD를 쓰면 학습이 단계적으로 진행됨.
        - 이 때, 각 단계는 1개의 Mini-batch ($x_{1...m}$ of size $m$) 를 의미함
            - Mini-Batch를 사용하면 얻는 이점
                1. Mini-batch에 대한 Gradient of Loss는 전체 Training Dataset에 대한 Gradient of Loss의 추정치임, 따라서 Batch size가 커질 수록 전체 Training Dataset을 더 잘 표현 할 수 있음
                2. Training set에 있는 Sample들을 각각 계산하는 것보다 Batch 단위로 계산하는게 더 빠름, 왜냐하면 GPU를 이용한 병렬계산이 잘되어 있기 때문임
- 층에 대한 입력이 바뀌게 되면 다음 층의 입력에 영향을 주고 층이 깊어질 수록 영향력이 증폭되어 학습이 더욱 복잡해짐
    
    $l = F_2(F_1(u,\Theta_1),\Theta_2)$
    
    $X = F_1(u,\Theta_1),\quad l=F_2(X,\Theta_2)$
    
    - $l$ = loss값,   $F_i$ = arbitray Transformations,   $u$ = input
    - $\Theta_2$의 관점에서 보면 $X$가 Input으로 들어오기 때문에 $X$의 값이 변하면 $\Theta_2$의 값도 변하게 됨
    - 앞서 말한 Internal Covariate Shift 문제임
    - 층의 입력이 바뀌면 학습이 복잡해지는 이유
        - 입력이 바뀌게 되면 layer의 Distribution이 계속 변경됨.
        - 변경된 새로운 Distribution에 layer가 또 다시 적응해야함
- 층에 대한 입력의 분포가 계속 고정된 상태로 유지 된다면, 입력이 거의 동일한 상태이기 때문에 다음 층이 재조정 될 필요가 없음
- 하위 layer들의 Parameter값을 변경 →  $|x|$ 의 값이 증가 → Activation Function(이 논문에선 Sigmiod Function)의 기울기가 0이 되는 지역으로 이동, 따라서 학습이 느려짐.
    
    ![기울기가 0이 되는 지점은 위의 빨간 박스이고, X대신에 Z라고 생각하면됨. 즉, 절대값 Z가 커디면 기울기가 0 이되는 지역으로 이동함](Batch%20Norm%2010e8f/Untitled%201.png)
    
    기울기가 0이 되는 지점은 위의 빨간 박스이고, X대신에 Z라고 생각하면됨. 즉, 절대값 Z가 커디면 기울기가 0 이되는 지역으로 이동함
    
    - Saturating현상과 Gradient Venishing이라고 하는 위 문제는 Activation Function으로 ReLU를 쓰거나, 낮은 LR을 사용하면 해결 할 수 있음
    - Non-linearity(Activation Function)의 입력 분포가 변화하지 않고 더 안정적으로 유지되면 Optimizer가 Saturating 지역으로 이동할 가능성도 줄어들고 훈련을 가속화 시킬 수 있음
- BN을 쓰면 얻는 장점
    - 높은 LR을 사용가능
    - Parameter 초기화를 덜 신경써도됨
    - Dropout을 사용안해도됨
    - 학습 속도가 빨라짐
    - Saturationg non-linearity를 사용할 수 있음
    - Saturated mode에 빠지는 것을 막음

# 2. Towards Reducing Internal Covariate Shift

- 층의 입력 분포가 고정시키고, 입력을 Whitening 시키면 모델이 더 빠르게 수렴하기 때문에 훈련 속도를 향상 시킬 수 있음
- 각 층의 입력을 Whitening 시키면 Internal Covariate Shift를 제거 할 수 있음.
    - Whitening이 가지는 문제점
        - Backpropagation 시에 Normalization하는 방식으로 Parameter를 업데이트 하기 때문에 Gradient step의 효과가 감소함
- Whitening이 학습에 미치는 악영향
    - Whitening은 Bias의 영향력을 없앰
        
        $X^`=W*X+Bias\\ E[X^`]= W*E[X] + Bias\\ X^{``} = X^`- E[X^`] \\ X^{``} = W*(X-E[X])$
        
    - Whitening은 Backpropagation과 무관하게 진행되기 때문에  Whitening을 통한 loss값의 변화가 없음
        - Whitening이 Bias를 없앴고, 이 결과는 Backpropa에 영향을 미치지 않기 때문에 Network가 계속 업데이트 되면서 새로운 Bias가 계속 더해짐, 결과적으로 계속 더해지기만 하는 Bias 때문에 Gradient가 발산하게 됨.
- 위의 Gradient Exploding 문제를 해결하기 위해 BN은 Normalization에 학습 가능한 Parameter를 넣어서 loss가 Normalization을 고려할 수 있도록 함.
- Norm layer의 Input을 Whitening 하는 것을 Backpropa에서 비용이 많이 듬
    - Network update를 통해서 parameter가 변경되면 Whitening을 위해서 Training Set에 대한 Covariate값을 계속 구해야하기 때문임

<aside>
📢 **따라서 모든 값에 미분가능(backpropa에 포함되기 위해)해야하고 Parameter Update 마다 training set에 대한 분석(Covariate값을 구하는 것)이 필요없는 Input norm을 찾아야함**

</aside>

# 3. Normalization via Mini-Batch Statistics

- 모든 곳에서 미분가능하지 않은 문제와 비용적 문제를 해결하기 위해 2가지 간소화를 진행

### 1) 각 Scalar Feature가 Mean=0, Variance=1을 가지도록 독립적으로 Normalize

- D개의 차원으로 구성된 Input을 각 차원에 대해서 Normalization 해줌
    
    ![Untitled](Batch%20Norm%2010e8f/Untitled%202.png)
    
    $\widehat{x}^{(k)} = \frac{x^{(k)}-E[x^{(k)}]} {\sqrt{Var[x^{(k)}]}}$
    
    - 전체 Training Set에 대한 각 Dimension 별 평균과 분산을 계산함
    - $\widehat{x}$ = Normalized x,   $E[X]$ = X의 평균,    $Var[X]$ = X의 분산,   $k$ = sample 번호
    - feature 간에 decorrelated(상관관계)가 없다면 수렴 속도를 높일 수 있음
    - 그냥 입력을 Norm하면 Saturating non-linearity의 경우에는 층의 Representation을 변경시키게됨
        - 예를 들어 Sigmoid의 입력을 Norm하면 비선형 속에 있는 선형지역으로 값이 제한됨
            
            ![Untitled](Batch%20Norm%2010e8f/Untitled%203.png)
            
- 위 문제를 해결하기 위해 Network에 삽입된 layer들은 Identity transform(항등변환)이 가능해야함
- 각 Activation $x^{(k)}$에 대해서 Scaling을 진행하는 $\gamma^{(k)}$값과 Shift를 진행하는 $\beta^{(k)}$값을 적용해줌
    
    $y^{(k)} = \gamma^{(k)}\widehat{x}^{(k)} + \beta^{(k)}$
    
    $\gamma^{(k)} = \sqrt{Var[x^{(k)}]}$
    
    $\beta^{(k)}=E[x^{(k)}]$
    
    - Identity transform을 위한 linear transform임, 따라서 Network의 출력을 복원하는데도 사용 가능
    - $\gamma^{(k)}, \beta^{(k)}$는 학습 가능한 Parameter임

### 2) 각 activation의 mean값과 variance값에 대한 추정치를 각 batch마다 생성

- SGD에선 mini-batch를 사용하기 때문에 가능
- 이렇게 하면 Normalization을 포함하여 Backpropa를 진행할 수 있음

![Untitled](Batch%20Norm%2010e8f/Untitled%204.png)

- 1개의 Batch에 대한 각 Dimenstion 별 평균과 분산을 계산함
    - 각각의 빨간박스로 $\gamma, \beta$값을 구할 수 있음

![Untitled](Batch%20Norm%2010e8f/Untitled%205.png)

- 정규화 후 다시 Scale & Shift를 통해 원래의 값으로 돌릴 수 있음.
    - Scale & Shift는 학습을 통해서 업데이트 되므로 전체 Training Set에 대해 작용하게 됨
- Mean=0, Variance=1이 유지되면서 전체 Training Set에 대한 Scale & Shift가 각각의 Input에 적용되기 때문에 다음 층의 Input이 되는 값은 항상 비슷한 분포를 유지하게 됨.
    - 따라서, 학습 속도가 빨라짐.
- BN은 학습 가능한 Affine Transform($y^{(k)} = \gamma^{(k)}\widehat{x}^{(k)} + \beta^{(k)}$)을 통해 Identity Transformation를 수행하고 Network Capacity를 보존함

## Training and Inference with Batch-Normalized Networks

- BN은 Activation Function 앞에 위치함
- BN을 사용하면 Batch size > 1 인 Batch Gradient Descent와 SGD를 사용할 수 있음
- 학습이 끝나고 Inference 때에는 Batch의 통계가 아니라 Population statistics를 통해 Norm을 수행
    
    $\widehat{x} = \frac{x-E[x]} {\sqrt{Var[x]}}$
    
    ![Untitled](Batch%20Norm%2010e8f/Untitled%206.png)
    
    - Inference 시에는 배치에 포함된 data 들을 이용하여 Mean과 Var를 구함
    - $\gamma,\beta$는 학습 동안에 얻은 값을 그대로 사용

## Batch Normalized Convolutional Networks

- Fully Connected 와 Convolutional layer 모두에서 BN을 사용 할 수 있음
- BN이 Conv layer의 특성을 따르기 위해선 1개의 Feature map 안에 있는 값들은 모두 같은 방법으로 Normalization 되야함
- Conv layer에 BN을 적용하기 위해서 1개의 Feature map에 대한 $\gamma,\beta$를 계산해야함
    - Batch size = M 이고 Feature Map size= p * q 이면, M * p * q 에 대해서 계산함
        - Algo 1. 에선 Scalar Feature에서 1개의 data가 1개의 차원을 의미함
        - Conv 에서 1개의 Feature map이 1개의 차원을 의미함.
- Conv layer 에선 feature map 당 $\gamma, \beta$ 를 계산 해야함
    - Algo 1, 2 모두 Feature map 당 계산하는 것으로 변경해주면 됨.

## Batch Normalization enables higher Learning rate

- BN은 layer Jacobians이 학습이 잘되도록 하는 Singular Values가 1에 가까워지도록 함

## Batch Normalization regularized the model

- BN을 사용하여 학습하게 되면 Batch 내부에 있는 example 끼리 결합되는 효과를 보여줌
    - 따라서 Model은 더 이상 Training set에 deterministic한 값을 생성하지 않음
    - **이 말은 BN이 모델의 Generalization에 이점을 준다는 것임**

# 4. Accelerating BN Networks

- 그냥 BN을 추가한다고 학습이 빨라지는건 아님

### BN의 모든 효과를 끌어내기 위해 해야하는 것

1. LR 증가
2. Dropout 제거
3. L2 weight regularization 감소
4. LR Decay 증가
5. Local Response Normalization 제거
    1. AlexNet에서 사용한 Normalization 방식
    2. 이젠 사용안함
6. Training set 더욱 철저하게 Shuffle
7. Photometric distortions(광도 왜곡) 감소

# 5. Conclusion

- BN을 통해서 Training 을 복잡하게 만드는 Covariate Shift를 제거함으로 학습 속도를 향상 시킴
- Normalization을 Network의 한 부분으로 만들어서 Optimizer에 의해 Update 될 수 있게 함
- 2개의 Parameter($\gamma,\beta$)만 추가하여 Network의 표현력을 그대로 유지함
- 높은 LR을 사용할 수 있고 Parameter 초기화에 덜 민감함
- Standardization layer와 목적은 유사하지만 다른 Task를 수행함
    - 차이점
        - Stand layer는 Activation function 후에 적용
        - BN은 Activation function 전에 적용
        - BN은 학습 가능한 $\gamma, \beta$를 가짐
        - BN은 Conv layer를 다룰 수 있음
        - BN은 Batch에 의존적이지 않은 Inference를 수행함
        - BN은 각 Conv layer 자체를 Normalization 함