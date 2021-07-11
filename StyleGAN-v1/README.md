# StyleGAN v1

### Paper

[https://arxiv.org/pdf/1812.04948.pdf](https://arxiv.org/pdf/1812.04948.pdf)

### Date

21-06-28 ~ 21-07-10

### R**eference**

[https://wiserloner.tistory.com/1199](https://wiserloner.tistory.com/1199)

[https://blog.promedius.ai/stylegan_1/](https://blog.promedius.ai/stylegan_1/)

[https://blog.promedius.ai/stylegan_2/](https://blog.promedius.ai/stylegan_2/)

[https://sensibilityit.tistory.com/510](https://sensibilityit.tistory.com/510)

[https://jayhey.github.io/deep learning/2019/01/16/style_based_GAN_2/](https://jayhey.github.io/deep%20learning/2019/01/16/style_based_GAN_2/)

---

- 들어가기 전 알아야 할 것. (내가 이해한 것이므로 확실하지 않음. 그냥 참고만 할 것.)
    1. **latent Space**
        1. Representation of compressed data, We can't interpret directly
        → Convolution layer를 통해 image를 그냥 압축시킨 것.
    2. **latent Code**
        1. image를 압축하여 추출된 특징(attribute)
        → Encoder를 통해서 압축하고 특징(attribute)를 추출함.
    3. **Entangle**
        1. 특징이 서로 얽혀 있는 상태여서 특징 구분이 어려움
    4. **Disentangle**
        1. 각 style들이 잘 구분되어 있어서 특징들이 분리가 잘 되어있음.
        → 어느 방향으로 가면 A라는 특징이 변함.
        2. 선형적으로 변수를 변경했을 때 어떤 결과물의 feature인지 예측 할 수 있음.

# 1. **Introduction**

- 2018년에 NVIDIA에서 발표한 논문.
- 현재 발표된 latent space Interpolation은 Generator끼리 비교할 수 있는 정량적 지표를 제공하지 않음 그래서 이 논문에선 Generator의 정량화를 위해 perceptual path length(지각경로길이)와 linear separability(선형 분리성)이라는 2가지 automated metrics를 제안함.
- Style Transfer에 사용하고 있는 Generator를 새로운 방법으로 재설계하여 영상 합성 과정을 제어 할 수 있게 함.
- 이 논문에서 소개하는 Generator는 학습된 상수를 Input으로 받아서 latent code를 기반으로 하는 각 convolution layer에서 image의 style을 조절함 그래서 다른 scale에서 image feature의 강도를 직접 제어 할 수 있음.
- Convolution layer를 통과한 style과 network에 직접 주입된 noise가 결합하는 구조는 생성된 image에서 Unsupervised Seperation of high-level attibute를 자동으로 수행하게 하고 직관적인 scale별 혼합과 interpolation을 가능하게 함.
- Input latent space는 training dataset의 확률 밀도를 따라하고 이것은 생성된 image에 불가피한 관여를 야기함 하지만 여기서 소개하는 Generator는 input latent code를 intermediate latent space에 박아 넣기 때문에 이러한 제한으로부터 자유로움.

---

# 2. Style-based Generator

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled.png)

- 여기서 소개하는 Generator 구조는 latent code를 input으로 받는 input layer를 생략하고 학습된 상수를 input으로 받음.
- latent code z(1x512 size)를 8-layer MLP로 구성된 mapping network를 통과시켜 latent space w(1x512 size)를 생성함.
- 생성된 w는 스타일을 뜻하는 vector code인데 이것을 layer 사이에 있는 AdaIN(Adaptive Instance Normalization)에 주입하면 각 layer에서 output에 style이 적용됨.
- w는 AdaIN에 주입되기 전에 shape을 맞추기 위해 Affine Transformation(그림의 A)을 적용함
- AdaIN은 Feature Map x (상수 + Noise 혹은 이전 layer output + Noise)와 style y를 입력으로 받음.
- 여기서 Style y의 차수가 2배가 된다는데 왜 2배가 되는지 모르겠음. Feature Map x를 scale하고 bias하기 때문이라는데.. 모르겠음.
- Style Transfer와 비교하기 위해 example image 대신에 mapping network에서 얻은 w를 style y로 사용함.
- 머리카락이 뻗치는 방향과 같은 무질서한 디테일을 위해서 Noise B는 Convolution layer 이후에 각 pixel에 더해지게됨 그래서 똑같은 image를 생성해도 머리카락의 방향은 조금씩 다름.
- 4x4부터 시작해서 각 해상도에는 2개의 convolution이 포함되기 때문에 총 9개의 해상도가 존재하고 18개의 convolution layer가 있고 AdaIN도 18개가 있음.

---

# 3. Quality of Generated Image

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%201.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%201.png)

- 결과값이 낮을수록 Generator의 성능이 좋음.
- CelebA-HQ dataset에는 WGAN-GP loss function을 사용했고 FFHQ dataset에서는 non-saturation loss with R1 regularization을 사용함.
- configuration (A)
    - Baseline model로 PGGAN(Progressive GAN)을 이용하였고 따로 명시되 있는 hyperparameter를 제외한 모든 hyperparameter를 상속받음.
- configuratrion (B)
    - Bilinear Upsampling / Downsampling을 추가함
    - training 시간을 늘리고 hyperparameter를 튜닝함. (디테일한 튜닝 hyperparameter는 부록 C에 적혀있음)
- configuration (C)
    - Mapping Network과 AdaIN을 추가함.
    - 위 2개를 추가하고 난 뒤에는 training 할 때 Input으로 latent code를 주입하여 얻는 이득이 사라짐
- configuration (D)
    - latent code를 Input으로 줘서 얻는 이득이 없기 때문에 Input layer를 제거하고 학습된 상수(4x4x512)를 Input으로 받음
- configuration (E)
    - Noise input을 추가함
    - 생성된 Image에 보다 세부적인 변화를 제어 할 수있게됨.
- configuration (F)
    - Mixing regularization을 추가함
    - 생성된 Image에 보다 세부적인 변화를 제어 할 수있게됨.

---

# 4. Properties of The Style-based Generator

- 특정 scale(convolution) 수정을 통해서 Generator가 생성된 image의 style을 제어 할 수 있음.
- Style 적용 강도를 수정할 수 있고 특정한 style만을 적용 할 수도 있음.
- 각 style의 효과는 network에 localization 되있기 때문에 style의 특정 부분을 수정하게 되면 생성되는 image의 특정 부분에만 영향을 줄 수 있음.
- AdaIN으로 각 convolution의 style을 제어하는데 이것은 각 convolution 마다 다른 style을 적용 할 수 있다는 말임.

---

# 5. Style Mixing

[https://youtu.be/kSLJriaOumA](https://youtu.be/kSLJriaOumA)

- Style을 지역화하기 위해 latent code z1, z2 2개를 운영함.
- z1 → w1, z2 → w2가 되고 w1는 생성되는 image의 장르(인종이나 나이를 의미하는듯)가 결정되기 전에 적용하고 w2는 그 이후에 적용함.
- w1, w2 를 이용하면 network가 특정 스타일끼리 상관 관계가 있다고 가정하는 것을 방지 할 수 있음.
→ ex) 대머리는 항상 선글라스를 끼고 있다면 network에선 대머리 == 선글라스라는 correlating이 발생함 그래서 대머리인 사람은 무조건 선글라스를 낀 사람으로 생성됨.
- 입력 image에 대한 latent code와 랜덤 값으로 구성된 latent code(혹은 다른 image에 대한 latent code)를 연산하여 mapping network에 주면 2개의 style을 mixing 할 수 있고 성능도 좋아짐.

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%202.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%202.png)

- 4x4 ~ 8x8 (빨강박스) - coarse(거친 혹은 조잡한) layer 혹은 coarse resolution 이로함
    - B에서 포즈, 헤어스타일, 얼굴모양, 안경 등을 가져오고 모든 색상(눈, 머리, 조명)과 미세한 얼굴 모양은 A를 닮음.
- 16x16  ~ 32x32 (파랑박스) - middle layer
    - B에서 얼굴 특징, 헤어스타일, 눈 감은것과 뜬것을 가져오고 포즈, 전체적 얼굴 모양, 안경 등은 A 를 닮음.
- 64x64 ~ 1024x1024 (노랑박스) - fine(미세한) layer 혹은 fine resolution
    - B에서 이 해상도를 복사하면 색과 미세 구조를 가져오고 나머지는 A의 스타일이 유지됨.

---

# 6. Stochastic variation

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%203.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%203.png)

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%204.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%204.png)

- 얼굴에는 머리카락의 방향이나 주근깨 등과 같이 Stochastic(확률론적)한 것들이 많음.
- [여기]()에서 말했다시피 이런 확률론적인 것들을 제어하기 위해 Convolution layer를 통과한 후 pixel마다 랜덤한 값으로 구성된 Noise B를 더해줌.
- Noise를 추가해야 무조건 정렬된 image만 생성하지 않고  무질서함이 포함된 디테일한 image를 생성할 수 있음 그리고 Noise도 학습됨.
- Figure 4는 같은 image의 다른 noise값을 추가했을 때 image의 큰 맥락은 유지되면서 디테일한 부분만 변화되는 것을 보여줌.
- Figure 5는 각 layer(혹은 해상도)마다 다른 noise 값을 추가 했을 때 image의 변화를 보여줌.
    - (a)
        - Noise를 모든 layer에 추가함
    - (b)
        - Noise를 적용하지 않음.
        - Noise를 생략하면 세세한 특징들이 사라짐.
    - (c)
        - Noise를 (64x64 ~ 1024x1024) 해상도에만 적용
        - 머리카락을 미세한 곱슬로 만들고 배경 디테일을 살림 그리고 피부 모공을 만듬.
    - (d)
        - Noise를 (4x4 ~ 32x32) 해상도에만 적용
        - 머리카락을 더 곱슬거리게 하고 배경을 더 크게 보이게함

---

# 7. Separation of Global Effects From Stochasticity

- Noise는 feature map의 각 pixel에 독립적으로 더해지기 때문에 image의 전체적인 맥락에 변화를 주지 않고 주근깨, 머리카락 등의 확률론적인 변화를 제어하는데 이상적임.
- Style 같은 경우 image의 전체적인 부분에 영향을 미침 왜냐하면 각 convolution 이후의 feature map들은 동일한 값(latent code w)으로 scale되고 bias 되기 때문.
- Noise로 pose와 같은 것을 제어하려 하면 공간적으로 일관성 없는 결정해서 Discriminator로부터 불이익을 받음.
→ why? 각 pixel 별로 독립적으로 연산되기 때문에 공간적인 정보가 포함안됨. 그래서 생성되는 image에 일관성이 없어지고 그렇게 되면 Discriminator가 fake로 인지함.

---

# 8. Disentanglement Studies

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%205.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%205.png)

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%206.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%206.png)

- Disentanglement 의 다양한 정의가 있지만 공통적인 목표는 하나의 변화요인(style을 말하는듯)을 제어하기 위한 latent space를 구성하는 것임.
- (a)같은 경우 실제 데이터 상의 분포를 표현한 것인데 보다시피 빈공간이 있음 이 부분은 데이터상으로 거의 나오지 않거나 관측하지 못한 부분임.
- (a)에서의 빈 부분을 매꾸기 위해 data를 억지로 encoding하여 한정되 공간 안에서 모두 표현하려하니 (b)와 같은 모습이 나옴.
- (b)같은 경우 그냥 encoding을 통과한 것이기 때문에 entangle(꼬였다) 이라고 함, 공간 내에서 값을 어느 방향으로 바꿔야 원하는 형태로 결과물을 얻을 수 있는지 알 수 없음.
- (b)에서 나온 latent space z(entangle된 vector 값)를 mapping 하여 disentangle된 (c) latent space w를 얻고, 이것을 가지고 원하는 형태로 결과물이 자연스럽게 변하도록 함.
- image를 encoding 하여 만들어진 z에는 image에 대한 모든 정보가 들어있음. 이 상태를 entangle한 상태라고 함.
- Mapping network는 entangle된 vector(latent space z)를 disentanlge된 vector(latent space w)로 만들어 주는 역할임.

---

# 9. Perceptual path length

- Mapping network를 통해서 entangle한 latent가 얼마만큼 disentangle한 latent로 변화했냐를 정량적으로 측정하는 지표. 즉, Mapping network에 대한 지표임.
- Mapping network의 입력으로 주는 latent space z의 값에 따라 latent space w가 얼마나 변하는지를 알기 위해 perceptually-based pairwise image distance를 사용함
- perceptually-based pairwise image distance 란?
    - latent space z에 아주 작은값 e(입실론)을 더해준 z+e와 그냥 z값을 각각 Generator에 넣어서  image ze와 image z를 얻음.
    - image ze와 z를 VGG16에 넣어서 나온 각각의 Feature값들의 distance를 구함
    → 이 방법은 image 간의 유사도를 거리로 구하는 방법임.
    - 이런 방법으로 z값의 변화에 따른 w값의 변화를 얻어 낼 수 있음.
    - 특정 부분에 대한 변화도를 알고싶다면 그 부분을 image z와 image ze에서 crop한 뒤에 사용하면 됨.

---

# 10. Linear Separability

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%207.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%207.png)

- latent space가 충분히 disentangle하면 한 가지 style만을 지속적으로 변화 시키는 방향 vector를 찾을 수 있어야함.
- 한 가지 style만을 지속적으로 변화 시킨다는 것을 정량화 하기 위해 linear hyperplane을 통해 latent space 점들이 2개로 얼마나 잘 분리 되는지 측정함.
→ ex) 한개의 point는 머리길이의 변화를 나타내고 다른 point는 나이 변화를 나타낸다고 했을때 충분히 disentangle하면 Linear hyperplane(linear SVM)으로 구분 할 수 있다는 말인듯.
- 원리가 이해 안됨. 잘 모르겠음 ㅎ.,ㅎ ㅈㅅ;

---

# 11. Truncation trick in W

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%208.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%208.png)

식. 1.

![StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%209.png](StyleGAN%20v1%20b3afd042d2d9449ea2cbd334a4382f30/Untitled%209.png)

식. 2.

- Training dataset 분포에서 밀도가 낮은 영역은 Generator가 학습하기 힘듬. 그래서 이 부분을 극복하기 위해 쓰는 방법임.
- 잘리거나 축소된 sampling space로부터 latent vector를 그리면 image의 질을 향상시킬 수있음.
- 식.1을 이용하여 z로부터 $\bar w$를 구하고 식.2 를 이용하여 $w^`$를 구하여 바로 Generator에 적용함.
- ψ == 1 일 때는 truncation trick을 사용하지 않은 것과 같음.
- ψ == 0 일 때는 모두 다 같은 얼굴이 나옴 즉 평균 얼굴이라 보면됨.

---

# 12. Conclusion

- Style-based Generator는 모든 방면에서 traditional GAN의 Generator보다 뛰어남.
- 특성에 대한 분리성, 확률적인 효과 그리고 중간 latent space의 선형성은 합성에 대한 GAN의 이해도와 제어성을 높일 수 있을 것임.
- 각 resolution 별로 image에 어떤 영향을 미치는지 알 수 있었음.
- 어떤 변수가 image에 어떤 style을 입히고 어떤 부분을 변화 시키는지 알 수 있었음.
- style mixing을 통해서 원하는 style들을 섞을 수 있고 style을 어느 해상도에 어떻게 적용하느냐에 따라 결과물이 달라진다는 것을 알 수 있었음.
- truncation traick을 통해서 조금 더 다양한 image들을 만들 수 있다는 걸 알았음.
- entangle과 disentangle의 개념을 잘 알 수 있었음.
- latent space와 latent code의 개념은 아직 헷갈림.
- 앞서 리뷰한 DCGAN도 어려웠지만 StyleGAN이 더 어려웠음 왜냐하면 비교적 최근 논문이고 선행 model인 PGGAN을 몰라서인듯.