# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

그 유명한  DCGAN 논문을 이제야 정리함. 완전 유명하니까 논문에서 읽은 내가 모르던 사실만 간략히 정리

## 1. Introduction

이 논문에서 크게 소개하는 바는 아래와 같다:

* DCGAN이라는 것을 제안하며 구조와 학습 방법 등을 소개
* 학습된 Discriminator(이하 D)를 image classification task에 feature extractor로 사용하므로써 기존의 unsupervised method와 비교
* DCGAN으로 학습괸 filter를 시각화해봄
* word embedding 같은 vector arithmetic property를 확인함

## 2. Related Work

다른 건 생략하고 간단히 정리하자면 unsupervised representation learning에는 지금가지 여러 연구가 진행되어 왔는데, 이런 연구들은 2가지 부류로 나눌 수 있다.

* parametric method

* non-parametric method

  ​

하지만 현재까지의 어떤 연구에서도(최근의 GAN까지도) 다음 2가지의 요건을 모두 충족하는 경우는 드물었다.

* 제대로 된 사진(즉, Blurry하거나 noisy하지 않은 사진)을 만드는 경우
* 이미 학습된 generator(이하 G)를 이용해 supervised tasks에 사용할 수 있는 경우



그리고 DCGAN은 위의 2가지 요건을 어느정도 모두 충족함



## 3. Approach and Model Architecture

현재까지 CNN과 GAN을 접목시키려는 많은 시도가 있었으나 성공적인 경우는 많지 않았다.

이 저자들도 위의 목적을 달성하기 위해 여러가지 시도를 하였고, 결론적으로 아래와 같은 시도들을 조합하니 결과가 좋았다고 한다.

1. G에 all convolution net을 사용하는 것
   * 이는 *Springenberg et al. 2014*를  참고하여 내려진 결정인데, deterministic spatial pooling function을 제거하므로써 G 스스로 자신에게 맞는 spatial upsampling을 배우도록 하기 위함이다.
2. 요즘 연구들의 트렌드를 따라 G와 D 모두에게 fully connected hidden layer를 제거하는 것
   * 이는 요즘 연구의 트렌드이기도 하지만 실험적으로 이러한 방법이 학습을 더 안정화 시켰다고 한다.
   * D의 경우, 마지막 convolution layer의 output을 flatten하여 sigmoid를 통과시킴
   * 참고로 이런 FC가 없는 것을 **global average pooling**이라고 부르는 듯
3. Batch normalization을 사용하는 것
   * Batch Norm은 초기화가 잘 안되서 생기는 문제를 해결해주고 gradient가 더 멀리까지 흘러갈 수 있게 도와준다.
   * 하지만 모든 layer에 Batch norm을 적용하면 오히려 학습이 불안정해졌는데, G의 output layer와 D의 input layer에는 Batch norm을 적용하지 않았더니 해결되었다.
4. Activation function의 경우, G는 output layer에 tanh를 사용하고 나머지에는 ReLU를 사용하였으며, D는 모두 leaky ReLU를 사용하는 것이 효과적이었다.



## 4. Details of Adversarial Training

* Image의 preprocessing:
  * 별도의 작업 없이 [-1, 1]로 정규화
  * 이는 G의 output layer activation function이 tanh이기 때문에 해줘야한다.
* Leaky ReLU의 alpha = 0.2
* SGD batch size = 128
* weights initialization = N(0, 0.2)
* AdamOptimization:
  * Learning rate = 0.0002
  * beta1 = 0.5

### Memorization과 Overfitting

DCGAN이 학습을 하던 중 그냥 사진을 외워서 만들거나 하는 경우(Memorization & overfitting)를 해결하기 위해 시도한 것은 다음과 같다.

먼저 3072-128-2072 짜리 Autoencoder를 학습시키고, 이 Autoencoder의 activation을 ReLU에 통과시켜 semantic hashing을 시켜 비슷한 사진들이 중복된 것을 제거(Deduplication).



## 5. Empirical Validation of DCGANs Capabilities

###  Classification CIFAR-10 Using GANs as a feature extractor

D를 feature extractor로 사용하여 사진을 분류하는 bench mark를 수행했더니 다른 것들보다 더 성능이 좋았다.

물론 Exemplar CNN같은 납땜질을 많이 한 것보다는 약간 성능이 안좋았지만 hyperparameter를 잘 튜닝하면 넘을거라고 생각하고 생략



## 6. Investigating and visualizing the Internals of the network

D의 activation을 조사해보면 feature들을 학습하였음을 알 수 있었고, Z(random noise)의 값을 변화시켜보면 만들어내는 사진들의 semantic이 변화하는 것을 관찰할 수 있음(예를 들면, 소파가 생겼다 사라졌다 같은 것)

#### Forgetting to draw certain objects

G가 만들어내는 사진들은 퀄리티가 좋기 때문에 이는 G가 어떠한 사실(예를 들면, 사진 속 Component나 object)을 배웠음을 알 수 있다. 이를 관찰하기 위해 간단한 실험을 해보면 기존의 conv layer에 창문에 반응하여 창문이 있으면 +, 없으면 -를 내보내는 filter가 있었는데, 사진에서 창문을 지운 뒤에 학습시켰더니 저 filter의 activation이 감소하는 (즉, -에 가까워지는) 현상이 있었으며, 실제로 만들어내는 사진에서 창문을 잘 그리지 못했다.



### Vector Arithmetic on Face Samples

이는 구글에서 발표한 word embedding 논문에서처럼 

> King - Man + Women = Queen

같은 것이 가능한지 알아본 것이며, 비슷한 사진 3개를 평균낸 경우 위와 같은 현상이 stable하게 나타났다.

예를 들면,

> Sunglass낀 남자 - 남자 + 여자 = Sunglass낀 여자

