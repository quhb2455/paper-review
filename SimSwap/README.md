# SimSwap

생성일: 2022년 2월 24일 오후 2:22

### Paper

[https://arxiv.org/pdf/2106.06340v1.pdf](https://arxiv.org/pdf/2106.06340v1.pdf)

### Date

22-02-21 ~ 22-02-24

### R**eference**

<aside>
📢 참고한 Blog가 없기 때문에 100% 내가 이해한 것 바탕으로 정리했음. 따라서 틀린 부분이 많이 있을 수 있음. 감안하고 읽기 바람.

</aside>

### 읽기 전 알아야 할 것들

- identity information
    - 사람의 신원을 알 수 있는 특징이나 정보를 말함
    - 예를 들어 박보영을 박보영이다 라고 인식할 수 있게 해주는 특징(눈, 코, 입의 생김새나 피부색)을 말함
- Attribute information
    - 사람의 얼굴이 가지는 특징
    - 예를 들어 표정이나 head position등을 말함
    

# Abstract

- 이전 연구들의 단점
    1. 임의의 얼굴에 대한 일반화 능력이 없었음
    2. 시선이나 얼굴 표정과 같은 특징들을 잘 살리지 못함
    
    ⇒ 이 2가지 단점들을 극복하기 위해 SimSwap을 만듬
    
- 위 단점을 해결하기 위한 2가지 접근
    1. `ID Injection Module`(`IIM`)을 사용하여 특정 얼굴에 대한 Face Swapping algorithm을 랜덤 얼굴에 대한 Face Swapping Algorithm으로 확장함
        - `ID Injection Module`(`IIM`) 이란?
            - Feature Level에서 Source 얼굴의 정보를 Target 얼굴로 전송해주는 모듈
    2. `Weak Feature Matching Loss`를 사용하여 facial attribute를 효율적으로 보존함

# 1. Introduction

- Face Swapping 의 가장 어려운 점 3가지
    1. 강력한 일반화 능력으로 임의의 얼굴에 잘 적용되어야함
    2. 결과 얼굴의 identity는 Source 얼굴의 identity와 유사해야함
    3. 결과 얼굴의 attribute와 Target 얼굴의 attribute(표정, 자세, 밝기 등등)가 일치해야함
- Face Swapping의 2가지 타입
    - **Source-oriented Method**
        - Image Level에서 작동
        - Target 얼굴의 자세와 표정 같은 특징(Attribute)들을 Source 얼굴로 전송한 후 섞음
        - 단점
            - Source 이미지의 자세와 밝기에 민감함
            - Target의 표정을 정확하게 반영하지 못함
    - **Target-oriented Method**
        - Feature Level에서 작동
        - Source 이미지의 변화를 Target에 잘 적용 할 수 있음
        - Target 이미지의 feature를 직접적으로 수정함
        - 대표 모델
            - OpenSource 모델 - `DeepFake`
                - Face Swapping을 통해 2개의 얼굴을 교환하여 새로운 얼굴 생성
                - 단점
                    - 일반화 능력이 떨어짐
            - GAN-Based 모델 - `Face Synthesis`
                - Source의 특징과  Target의 특징을 Feature level에서 합쳐서 임의의 얼굴을 생성
                    
                    ⇒ Source와 Target이 섞인 얼굴
                    
            - 최근 모델 - `FaceShifter`
                - 2-stage framework을 사용하여 높은 정확도의 결과물 생성
            
            ⇒ 위의 모델들은 Identity 정보에 너무 집중하여 얼굴표정이나 자세를 잘 보존하지 못함
            
- 본 논문에선 Identity 정보도 잘 활용하고 Attribute 정보도 잘 보존하는 방법을 제시
    - `**ID Injection model**`
        - 일반화 능력(Generalization) 부족을 극복하기 위해 만듬
        - Source 얼굴의 Identity 정보를 Embedding 하여 Target 얼굴의 feature를 수정함
            - Identity 정보와 Attribute 정보를 Feature Level에서 결합시킴
        - 임의의 얼굴에 적용할 수 있음
        - Identity 정보와 Decoder간의 관련성을 제거함
            - 일반화 능력 상승을 위하여
    - `**Weak Feature Matching Loss**`
        - 생성된 결과 이미지의 각 Attribute들을 Target 이미지에 매치시키는 복잡한 과정을 해결하기 위해 만듬
        - Semantic Level에서 결과 이미지가 Target 이미지의 Attribute 속성을 잘 보존하도록 도와줌
        - 이 Loss function을 통해 이전 Method들 보다 더 나은 Attribute 보존 기능을 갖게 됨

# 2. Related Work

## 1) Source - Oriented Methods

- Target 얼굴의 Attribute를 Source 얼굴로 전송해서 섞는 방법
- 대표 모델들
    - Face Change Model(2014)
        - 3D model을 사용해서 Target의 자세와 밝기 등을 Source로 전송함
        - 단점
            - 수동 조작이 필요함
    - Face Swapping Model(2018)
        - 단점
            - 모델이 보유한 얼굴 목록 안에서만 Swapping이 가능함
            - Identity 정보만 Swapping 됨
    - Face Segmentation & Swapping & Reception Model(2018)
        - 3D Face Dataset을 활용
        - 자세와 표정을 Poisson Blending을 사용하여 Source와 섞음
        - 단점
            - 3D Face Dataset 안에 포함되어 있는 표정들만 처리가능
    - FSGAN(2019)
        - 2 - stage 로 이루어짐
        - Face reenactment model을 이용하여 표정과 자세를 Source로 전송함
        - Face Inpainting model을 이용하여 Source와 Target을 섞음
        - 단점
            - Source 얼굴의 과장된 표정이나 큰 동작들이 결과에 큰 영향을 줌

## 2) Target - Oriented Methods

- 동작 3단계
    1. Neural Network를 이용하여 Target에서 Feature값을 추출
    2. Featuret값을 수정
    3. 수정된 Feature 값으로 Face Swapping 된 새로운 이미지 생성
- 대표 모델들
    - Face Swapping using CNN(2017)
        - Generator를 사용함
        - 단점
            - 특정 1개의 얼굴만 바꿀수 있음
    - DeepFake(2020)
        - Encoder-Decoder 구조를 사용
        - 한번 학습 시키면 2개의 특정 identity 간의 Face Swapping이 가능함
        - 단점
            - 일반화에 약함
    - FSNet(2018) & RSGAN(2018)
        - Source의 얼굴과 Target의 얼굴이 아닌 부분에서 각각의 Latent representation을 합쳐서 새로운 얼굴을 생성
        - 단점
            - Target 얼굴의 표정을 유지 할 수 없음
    - IPGAN(2018)
        - Source 이미지로부터 identity 벡터를 추출하고 Target 이미지로부터 Attribute 벡터를 추출해서 Decoder로 전송하여 이미지 생성
        - 생성된 이미지는 Source 이미지의 identity 정보가 잘 보존되어 있음
        - 단점
            - Target 얼굴의 표정이나 자세를 잘 보존하지 못함
    - FaceShifter(2019)
        - Face Swapping을 했을 때 결과가 꽤 좋음
        - 단점
            - 위에서 소개한 model 들과 마찬가지로 Attribute에 너무 약한 제약을 가해서 표정이 잘 매치되지 않음

# 3. Method

- 본 논문에서 소개하는 Framework는 Target 얼굴의 Attribute를 변경하지 않고 Source 얼굴을 Target 얼굴로 잘 전송시켜주는 Framework임
- Identity-Specific Face Swapping에서부터 확장시켜서 임의의 얼굴에도 잘 적용할 수 있도록 함

## 1) Limitation of the DeepFakes

- 기본 구조
    
    ![DeepFake 논문이 없어서 구글에서 사진을 긁어 왔음](SimSwap%206ee0d/Untitled.png)
    
    DeepFake 논문이 없어서 구글에서 사진을 긁어 왔음
    
    - Encoder - Decoder(source)
        - Encoder를 통해 source 이미지를 Warpping하고 Decoder를 통해 다시 원상 복구 시킴
    - Encoder - Decoder(target)
        - Encoder를 통해 Target 이미지를 Warpping하고 Decoder를 통해 다시 원상 복구 시킴
    
    ⇒ Decoder가 2개 있는  AutoEncoder 구조임
    
- Test 시 사용되는 구조와 역할
    - Target이미지를 Encoder - Decoder(source)에 넣으면 Decoder가 Target이미지의 Attribute는 유지한채 Source 이미지의 identity를 입혀줌
    - Encoder는 Target 얼굴의 identity와 attribute를 모두 포함한 feature 값을 추출함
    - Decoder(source)는 Target feature를 Source 이미지의 identity를 가진 이미지로 변환함
        - 이 때, source 이미지의 identity는 Decoder(source)의 weight에 녹아있음
        ⇒ Why? Reconstruction으로 Sorce 이미지를 복구 시키는 것을 학습 했기 때문에 Source 이미지에 대한 Identity 정보를 Decoder의 Weight가 가지고 있음
    
    <aside>
    ⚠️ 위에서 설명한 구조와 각 구조가 동작하는 방식을 보면 DeepFake는 특정한 1개의 identity에 대해서만 동작할 수 있음
    
    </aside>
    

## 2) Generalization to Arbitrary Identity

![Untitled](SimSwap%206ee0d/Untitled%201.png)

- 위와 같은 문제점들을 극복할 수 있는 방법은 Identity 정보를 Decoder로부터 분리하는 것임
    - 분리하게 되면 Decoder가 1개의 이미지에 대한 Weight로 구성되지 않음, 따라서 전체적으로 일반화 능력이 상승하게 됨
- Identity정보를 Decoder와 분리시키기 위해서 `ID Injection Module`(이하 `IIM`)을 Encoder와 Decoder 사이에 배치함
- Target 이미지의 Attribute를 보존하고 Identity만 변경하기 위해서 Training Loss를 사용함
    - Training Loss를 통해서 Network가 학습을 통해 Target 이미지의 어떤 부분을 보존하고 어떤 부분은 바뀌어야하는지 알 수 있음
- IIM을 통해서 Target의 Identity를 Source의 Identity로 변경시켜줌
- IIM의 구조
    - Identity Extraction Part
        - Source 이미지가 가지고 있는 Identity 벡터를 추출함
            - 추출된 Identity는 ID Embedding과 ID Loss에 사용됨
        - Result 이미지로부터 Source 이미지의 Identity 벡터를 얻기위해 Face Recognition Network를 사용함
            - Result 이미지로부터 추출된 Source이미지의 Identity 벡터는 ID Loss에 사용됨
        
        ⇒ 2개의 방식으로 추출된 Source 이미지의 Identity 값으로 ID Loss를 구하고 Source 이미지의 Identity값이 계속 유지되도록함
        
    - Embedding Part
        - `ID Block`을 이용하여 Source 이미지로부터 추출된 Identity 벡터를 Target Feature에 주입함
            - Target Feature는 Encoder를 통과해서 나온 Feature값임
        - ID Block은 Residual Block을 조금 수정한 것
        - 본 논문에서 9개의 Block을 사용
        - ID Block에 `Adaptive Instance Normalization`(`AdaIN`)을 사용
            
            ⇒ AdaIN은 StyleGAN에서도 사용함
            
- IIM에 있는 ID Block을 통해서 Identity 값이 Feature에 주입된 후에 Decoder를 통과시켜 이미지를 생성함
- Decoder는 입력된 이미지(Target)을 복원시키는 쪽으로 학습이 되기 때문에 특정 Identity와는 관련이 없음
    - Identity 추출과 입력이 되는 부분은 모두 IIM에 있는 ID Block에서 일어나기 때문에 Decoder와 전혀 관련이 없음
- 학습 중에는 생성된 Result 이미지의 Identity 벡터와 Source 이미지의 Identity 벡터 간에 Loss를 최소화 하는 방향으로 진행됨
    - `Identity Loss`를 사용하여 최소화함
        - Identity Loss에는 문제점이 있음
            - Network를 Overfitting 시킴
            - Target 얼굴의 Attribute는 없어지고 Source 얼굴의 Idenetity만 포함되어 있는 정면 얼굴을 생산함
            - 해결책
                - PatchGAN의 Discriminator를 사용
                    - 잘못된 이미지를 구별하기 위해
                - Adversarial Loss 사용
                    - 생성된 이미지의 퀄리티를 높히기 위해

## 3) Perserving the Attribute of the Target

- 현재는 Target의 Identity와 attribute를 모두 포함한 Target Feature를 직접적으로 수정하기 때문에 Attribute만 유지한채 Identity를 수정하기 어려움
- 위 문제점을 해결하기 위해서 Training Loss를 이용하여 Identity와 Attribute에 제약을 걸어야함(2개를 분리해야함)
    - 명시적(직접적)으로 모든 Attribute를 제약하게되면 1개의 Attribute에 대해 1개의 Network가 필요함
    

### **Weak Feature Matching  Loss**

![Untitled](SimSwap%206ee0d/Untitled%202.png)

![Untitled](SimSwap%206ee0d/Untitled%203.png)

- 암묵적(간접적)으로 제약을 걸기위해 사용
- Pix2PixHD의 Feature Match Loss를 활용함
    - Pix2PixHD는 Ground Truth Image와 Generated Image로 부터 각각 feature를 추출하는 Discriminator 2개 를 사용함
    - 해당 Discriminator들의 각 Layer가 추출한 Feature 값을 비교하는 Loss Function이 `Feature Matching Loss`임
    - `**Weak Feature Matching Loss**`는 Feature Match Loss처럼 모든 Layer를 이용하는 것이 아니라 하위 layer 몇 개만 이용함
        - 출력에 가까운 하위 layer만 이용하는 이유는 입력과 가까운 layer는 Attribute 정보와 관련되어 있기 때문
        - 주로 Idenetity에 관련된 값이 있는 하위 layer에 대해서만 Feature Matching을 진행함으로써 간접적으로 Attribute와 Identity에 대해서 제약을 걸 수 있음
        ⇒ 간접적으로나마 분리 시킬 수 있음
- `Original Feature Match Loss` 수식
    
    $L_{oFM}(D) = \sum_{i=1}^M\frac{1}{N_i}\parallel D^{(i)}(I_R) - D^{(i)}(I_{GT})\parallel_1$
    
    - $M$ = 전체 layer 개수
    - $N_i$ = $i$번재 layer
    - $D^{(i)}$ = Disciriminator의 $i$번째 layer
    - $D$ = Discriminator
    - $I_R$ = Generated output Image
    - $I_{GT}$ = Ground Truth Image
- 본 논문에서 사용한 `Weak Feature Matching Loss` 수식
    
    $L_{wFM}(D) = \sum_{i=m}^M\frac{1}{N_i}\parallel D^{(i)}(I_R) - D^{(i)}(I_T)\parallel_1$
    
    - $m$ = Weak Feature Matching Loss 계산을 시작할 Layer 번호
    - $I_T$ = Target Image
        - Face Swapping에는 GT가 없기 때문에 Target image를 GT 대신에 사용
- Original과 `Weak Feature Matching Loss`의 차이점
    - Original은 학습의 안정성을 위해 사용
    - Weak Feature Matching Loss 는 Identity와 Attribute를 구분하기 위해 사용
    

### **Overall Loss Function**

- SimSwap은 5개의 loss function을 사용하고 있음
    1. Identity Loss
        - Result Image의 identity 벡터와 Source Image의 identity 벡터간의 거리를 계산
        - $L_{Id} = 1 - \frac {v_R \cdot v_S} {\parallel v_R\parallel_2\parallel v_S\parallel_2}$
    2. Reconstruction Loss
        - Result Image가 Target Image(input)로 유사해지기 위해 사용
        - $L_{Recon} = \parallel I_R-I_T\parallel_1$
    3. Adversarial Loss 
        - 큰 자세를 위해 Multi-scale Discriminator 사용
    4. Gradient Penalty
        - Discriminator가 발산하는 것을 막기 위해 사용
    5. Weak Feature Matching Loss
        - Identity와 Attribute를 구분하기 위해 사용
- 전체 수식
    - $\lambda_{ID}L_{ID} +\lambda_{Recon}L_{Recon}+L_{Adv}+\lambda_{GP}L_{GP}+\lambda_{wFM}L_{wFM\_sum}$
    - $\lambda_{Id} = 10$
    - $\lambda_{Recon}=10$
    - $\lambda_{GP}=10^{-5}$
    - $\lambda_{wFM}=10$
    

# 4. Experiments

- Implementation Detail
    - large face dataset VGGFace2 사용
        - 250x250 사이즈 이미지는 삭제
    - align and crop 은 224 사용
    - Pretrained Arcface model 사용
    - Adam optimizer 사용
    - Identity가 같은 이미지는 한 배치에 넣고 Identity가 다른 이미지는 또 다른 배치에 넣어서 번갈아가면서 훈련
        - 뭔말임???
    - 500 Epochs
- 과장된 표정, 얼굴에 있는 줄무늬, 옆모승 등을 잘 보존함.
    
    ![Untitled](SimSwap%206ee0d/Untitled%204.png)
    

## 1) Comparison with Other Methods

![Untitled](SimSwap%206ee0d/Untitled%205.png)

![Untitled](SimSwap%206ee0d/Untitled%206.png)

![Untitled](SimSwap%206ee0d/Untitled%207.png)

- 비교에 사용한 데이터셋
    - FaceForensics++
        - 1000개의 얼굴 비디오와 1000개의 deepfake로 face swapping된 얼굴 비디오로 이루어져 있음
- figure 4를 이용한 비교
    - `Deepfake` 는 밝기와 자세가 잘 안맞는 것을 볼 수 있음
    - `FaceShifter` 생성 결과는 좋으나 표정이나 시선이 제대로 처리가 안된 것을 볼 수 있음
        - Identity 정보에 너무 많이 집중해서 Attribute정보가 유지되지 못하여 발생하는 문제임
- figure 6을 이용한 비교
    - `FSGAN`는 Source 얼굴에 민감하고 시선방향과 표정을 제대로 생성하지 못함
    - FSGAN은 조명 조건에 따라 결과 이미지와 입력 이미지가 다름
    ⇒ 사진으로 봐선 모르겠는데 논문에서 그렇다고함;

## 2) Analysis of SimSwap

![Untitled](SimSwap%206ee0d/Untitled%208.png)

- ID retrieval 이라는 것을 사용해서 점수를 매김
    - 모델의 identity 관련된 수행능력을 평가하는 것인듯
- SimSwap-oFM 은 Original Feature Matching을 썼다는 것이고 nFM은 No Feature Matching을 썼다는 것임
- ID점수는 FaceShifter가 높지만 Posture Distance를 봤을 땐 SimSwap-oFM 값이 가장 낮으므로 Posture를 가장 잘 표현했다고 볼 수 있음
- 위 figure 4, figure 5를 보면 Identity performance는 FaceShifter보다 떨어지지만 Attribute는 더 잘 보존할 수 있음

### Keeping a Balance Between Identity and Attribute

- IIM은 identity를 Embedding 할 때 불가피하게 Attribute 보존 능력에 영향을 끼치기 때문에 identity와 Attribute 사이에서 균형을 잘 맞춰야함
- SimSwap에서 Identity와 Attritbute의 균형을 맞추는 2가지 방법
    1. $\lambda_{ID}$ 의 값을 높혀서 조금 더 Identity를 강하게 수정 할 수 있도록 함
    2. Feature Matching을 할 때 feature의 수를 조절하는 것
    
    ⇒ 위 2가지 방법을 섞어서 더 좋은 결과물들을 만들 수 있음
    
    ![Untitled](SimSwap%206ee0d/Untitled%209.png)
    
    - SimSwap 기준 왼쪽에 있는 것들은 ID Retrieval 점수가 SimSwap보다 낮음
        
        ⇒ SimSwap보다 Identity 능력이 안 좋음
        
        ⇒ 왼쪽에 있는 Method들은 Weak Feature matching을 하는게 성능이 더 좋다는 것을 반증해주고 있음. 왜냐하면 SimSwap 보다 ID Retrieval이 낮으니까 
        
    - nFM은 ID Retrieval이 가장 높지만 Recon loss도 가장 높음
        
        ⇒ Identity 수행능력을 너무 높혀서 Attribute를 보존하질 못함
        
    - SimSwap이 ID Retrieval도 높고 Recon loss 도 나름 중간에 속하기 때문에 Identity와 Attribute의 balance가 잘 맞다고 좋다고 볼 수 있음
    

# 5. Conclusion

- SimSwap은 강력한 일반화와 높은 정확도에 집중한 Face Swapping Framework임
- ID Injection Module은 Identity 정보를 Feature level에서 수정하고 identity-specific face swapping을 arbitrary face swapping으로 확장시킴
- Weak Feature Matching Loss는 framework가 더 좋은 Attribute 유지 능력을 가질 수 있게 함
- 이전 method들 보다 더 좋은 결과를 생성할 수 있고 Atrribute 보존 능력이 좋음