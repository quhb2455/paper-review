# DCGAN

### Paper

[https://arxiv.org/pdf/1511.06434.pdf](https://arxiv.org/pdf/1511.06434.pdf)

### Date

21-05-06 ~ 21-05-16

### R**eference**

[https://angrypark.github.io/generative models/paper review/DCGAN-paper-reading/](https://angrypark.github.io/generative%20models/paper%20review/DCGAN-paper-reading/)

[http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html](http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html)

[https://m.blog.naver.com/laonple/221201915691](https://m.blog.naver.com/laonple/221201915691)

[https://arxiv.org/pdf/1412.6806.pdf](https://arxiv.org/pdf/1412.6806.pdf)

[https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)

---

# 1. **Introduction**

- 대부분의 setting에서 학습이 안정적으로 이뤄지는 Convolutional GAN의 구조를 제안하고 검증 함.
- 학습된 Discriminator로 classification 문제를 수행하여 다른 비지도 학습 알고리즘과 성능 비교를 함.
- 학습된 filter를 시각화하여 특정 filter가 특정 object를 그린다는 것을 보여줌.
- Generator가 생성하려는 sample이 vector arithmetic properties(벡터산술특징)을 가져서 벡터 산술 연산으로 쉽게 조작 될 수있음을 보여줌.

---

# 2. Related Work

## 1) Representation Learning From Unlabled data

- 비지도 학습의 전통적인 방법은 data에 clustering을 하여 Classification score를 올릴 수 있음.
- image를 compact code로 encode하는 AutoEncoder를 학습하는 방법이 있음.

## 2) Generating Natural Images

- Generator model은 Parametric 과 non-Parametric으로 분류되는데 DCGAN은 Parametric에 포함됨.
- 자연스러운 image를 생성하는 것은 지금까지 좋은 성과를 거두지 못함.
- Variational Sampling Approach는 image를 흐릿하게 생성함.
- GAN은 의미없는(?) 이상한 image를 생성하였고, 생성된 image애눈 noisy가 많았음.
- Laplacian Pyramid Extension은 높은 퀄리티의 image를 생성했지만 여러 개의 model들을 연결 시켰었고 그로인해 noisy가 발생하고 객체가 불안정함.
- Recurrent Network와 Deconvolution Network는 자연스러운 image를 생성 했지만 Generator를 지도학습에 이용하지 않았음.

## 3) Visualizing the Internals of CNNs

- Neural Network는 Black box라서 Network에서 무슨 일이 발생하는 지 알 수 없음.
- Network 내부에서 무슨 일이 발생하는지 알기 위해 Deconvolution과 최대로 활성화된 filter를 사용하여 각각의 convolution filter가 어떤 목표(?) 혹은 객체를 만드는지 대략적으로 보여줄 수 있음.

---

# 3. Approach and Model Architecture

- 이전에도 Convolution을 GAN에 넣으려는 시도가 많았는데 Scale up 문제 때문에 대부분 실패 했음.
- LAPGAN이 높은 품질의 image를 생성하기 했지만 객체가 흔들려보이는 문제가 있었음.
- 이 논문에서 설명하는 구조는 꽤나 높은 품질의 image를 single shot으로 만들 수 있고 다양한 DB에서 안정적인 학습이 가능함.
- CNN 구조에서 3가지 변경 사항을 적용했음.

## 1) Architecture Guideline

![DCGAN%2069548870c8674d00a338d79ecd448827/fractional_padding_strides_transposed.gif](DCGAN%2069548870c8674d00a338d79ecd448827/fractional_padding_strides_transposed.gif)

<Fractional-Strided Convolution>

![DCGAN%2069548870c8674d00a338d79ecd448827/padding_strides.gif](DCGAN%2069548870c8674d00a338d79ecd448827/padding_strides.gif)

<Strided Convolution>

- Pooling Layer를 Strided Convolution(for Discriminator)이나 Fractinoal-strided Convolution(for Generator)로 대체하여 All Convolution Net이 되게함.

    Strided Convolution이 되면 network 자체로 Downsampling을 학습할 수 있게 됨.
    Fractional-strided Convolution이 되면 network자체로 Upsampling을 학습 할 수 있게됨.

- Generator와 Discriminator에 BatchNormalization(이하 BN)을 사용하는데 모든 layer에 적용하면 sample이 흔들리고 model 불안정해지기 때문에 Generator의 output layer와 Discriminator의 input layer에는 적용하지 않음.

    BN을 사용하면 GAN에서 공통적 실패 요인인 collapsing문제를 어느정도 해결 할 수 있음.

- Fully-Connected Layer를 삭제하는데 그 이유는 Global average pooling을 적용하여 image classification SOTA를 달성한 [이 논문](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)에서 알려줌. (현재 읽어보지 않아서 모름..)
- Generator에서 output에 적요되는 Tanh을 제외하고 모든 layer에 ReLu activation을 적용함.
- Discriminator에서 모든 layer에 Leaky-ReLu activation을 적용함.

![DCGAN%2069548870c8674d00a338d79ecd448827/generator.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/generator.jpg)

![DCGAN%2069548870c8674d00a338d79ecd448827/discriminator.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/discriminator.jpg)

# 4. Details Of Adversarial Training

- LSUN(Large-Scale Scene Understanding), ImageNet-1K 그리고 Face dataset을 이용했음.
- Tanh activation의 범위를 [-1, 1]로 조절한 것 외에는 Image에 어떠한 preprocessing도 적용하지 않음.
- LSUN dataset을 통해서 DCGAN model이 large data와 고해상도 image에 대해 어떻게 scaling 하는지 보여줌.
- DCGAN이 dataset에 overfitting되서 sample을 암기하는 형식으로 고품질 image를 생성해내는게 아니라는 것을 입증하기위해 수렴 후 생성된 image와 1 epoch 후 image를 비교 분석함.

![DCGAN%2069548870c8674d00a338d79ecd448827/1E.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/1E.jpg)

1E 후 생성한 image

- 작은 learning rate와 작은 batch size를 사용했고 1E 만 돌았기 때문에  overfitting 가능성은 낮음.

![DCGAN%2069548870c8674d00a338d79ecd448827/5E.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/5E.jpg)

5E 후 생성한 image

- 반복되어 나타는 noisy(흰색 반점)들이 Underfitting 되어 있는 증거임.
- 1E 때보다 image의 품질이 더 나아진것을 볼 수 있음.

## 1) Walking On The Latent Space

![DCGAN%2069548870c8674d00a338d79ecd448827/waling_on_the_latent_space.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/waling_on_the_latent_space.jpg)

- latent space가 의미있는 결과를 생성한다면 model이 제대로 학습 했다고 할 수 있음.
- 위 사진에서 볼 수 있듯이 latent 변수 z를 조금씩 변화시켜주면 TV가 창문으로 변하는 것을 볼 수 있음.

## 2) Visualizing The Discriminator Feature

![DCGAN%2069548870c8674d00a338d79ecd448827/discriminator_feaure.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/discriminator_feaure.jpg)

- random 값으로 초기화된 filter 는 어떤것에서도 활성화 되지 않음.
- [이 논문](https://arxiv.org/pdf/1412.6806.pdf) 에서 제안한 guided backpropagation을 사용하면 filter가 침대와 창문 같은 침실의 일반적인 부분을 학습한 것을 알 수 있음.

## 3) Forgetting To Draw Certain Object

![DCGAN%2069548870c8674d00a338d79ecd448827/draw_window.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/draw_window.jpg)

- Generator는 scene에서 특별한 객체들을 학습하는데 어떤 객체들을 학습하는지 보기 위해 생성기에서 나오는 창문을 제거함.
- 창문 위에 bbox가 그려진 data로 학습하고 학습중에 창문 부분의 weight가 양수이면 weight를 drop시켰음.
- Network에서 창문을 그리지 않고 다른 객체로 그 부분을 채우는 것을 볼 수있음

## 4) Vector Arithmetic

![DCGAN%2069548870c8674d00a338d79ecd448827/vector_arithmetic.jpg](DCGAN%2069548870c8674d00a338d79ecd448827/vector_arithmetic.jpg)

- DCGAN 개발자들은 학습을 통해 얻어진 z 값으로 vector arithmetic(벡터연산)이 가능 하다는 걸 알아냄
- z 값 중 "안경을 쓴 남자" 를 그리게 하는 입력값의 평균치와 "안경이 없는 남자"와 "안경이 없는 여자"의 평균치도 구해서 각각 빼고 더해주어서 얻은 z' 값을 Generator에 넣게 되면 "안경을 쓴 여자" image를 생성해냄.

---

# 5. Conclusion

- 어떠한 이론으로 알아낸 구조가 아니고 반복된 시도로 알아낸 구조임.
- 반복된 시도로 알아낸 구조 임에도 불과하고 뛰어난 성능을 보여줌.
- 각 필터가 어떤 객체를 그리는지도 알아 낼 수 있었고 latent variable z가 벡터연산이 가능하다는 것도 알아냄.
- Convolution을 GAN에 적용하여 이후 나오는GAN에 큰 영향을 미치고 여기서 나온 구조가 거의 default로 활용됨.
- GAN을 처음 접하는 나에겐 내용이 조금 어려웠음.
- 이 논문에 설명되어 있다! 이렇게 쓰여진 부분이 있어서 이해하는데 시간이 오래 걸렸음. 안타깝게 링크가 걸린 논문은 읽어보지 못함.