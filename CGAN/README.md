# cGAN

### Paper

[https://arxiv.org/abs/1411.1784](https://arxiv.org/abs/1411.1784)

### Date

21-05-06 ~ 21-05-16

---

# 1. **Introduction**

- GAN의 Conditional version으로 label data Y를 generator와 discriminator에 적용하는 구조.
- Unconditional Generative model은 data 생성을 통제 할 수 없음.
- Conditional Generative model은 추가적인 정보를 통해 data 생성을 통제 할 수 있음.
- 2014년에 발표된 논문이고 DCGAN 이전에 나왔음.

---

# 2. Related Work

## 1) Multi-modal Learning For Image Labelling

- 지도학습은 큰 성공을 거뒀지만 여전히 많은 수의 class를 예측하는 것은 어려움.
    - 해결법 : 다른 modality에 대한 정보를 활용하는 것으로 해결 가능함.
    ex) 단어에 대한 vector representation을 학습하기 위해 자연어 말뭉치를 사용하는 것.
- 일상 생활의 많은 data는 1:대 매칭이지만 최근 연구들은 input-output을 1:1 매칭에만 집중함.
    - 해결법 : 1:1매칭 문제는 conditional probabilistic generative model (조건부 확률적 생성 모델)을 사용 함으로써 어느정도 해결 가능.

---

# 3. Conditional Adversarial Nets

## 1) Generative Adversarial Nets(GAN)

- Generator(생성기) : 학습 data 분포와 유사한 data 분포를 만들어서 data를 생성해는 model.
- Discriminator(판별기) : Generator가 생성한 data가 진짠지 가짠지 판별해주는 model.
- GAN의 Loss 함수에 관한 설명은 [이 곳](https://www.notion.so/1-GAN-f482cbbdbf054c05ba990458946012d0) 참고.

![cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled.png](cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled.png)

## 2) Conditional Adversarial Nets

- Class label과 같은 추가적인 정보 Y를 추가하면 GAN을 Conditional model로 확장 할 수 있음.
- Y를 Discriminator 와 Generator의 input에 추가하면 Conditional model이 됨.

![cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%201.png](cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%201.png)

- cGAN의 Loss함수는 아래와 같고 GAN Loss 함수 input에 조건y가 추가된 조건부 형태임.

![cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%202.png](cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%202.png)

---

# 4. Experimental Results

## 1) Unimodal

- Generator model
    - noise z는 Uniform distribution에서 추출했고 size = 100 임.
    - z와 y는 layer size 각각 200, 1000인 hidden layer(ReLu)에 매핑됨.
    - z와 y는 차원이 1200인 hidden ReLu layer로 합쳐짐.
    - 마지막엔 sigmoid unit을 사용하여 784차원(MNIST = 28*28)으로 변환
- Discriminator
    - x는 240 unit과 5piece 짜리 maxout layer로 매핑하고 
    y는 50 unit과 5piece 짜리 maxout layer로 매핑함.
    - 위의 두 layer는 sigmoid로 들어가기 전에 240unit과 4piece 짜리 maxout layer로 매핑됨.

![cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%203.png](cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%203.png)

- CGAN은 다른 network와 base가 비슷하지만 성능은 non - Conditional GAN을 포함해도 상위에 속함.
- hyperparameter와 model structure를 조금 더 연구하면 성능 개선을 할 수 있음.

## 2) Multimodal

- 1개의 사진(조건 Y)에 대해 User-generated metadata + Annotation (data X) 사용한 label generator model을 만듬.

![cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%204.png](cGAN%20eb884786ddf148cfa867466fbba6be6c/Untitled%204.png)

---

# 5. Conclusion

- Image Generate 성능은 original GAN과 비슷하지만 output을 조절할 수 있다는 점에서 의미가 있음.
- Pix2Pix와 CycleGAN에 영향을 끼침.