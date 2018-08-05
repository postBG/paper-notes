# [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)

## Abstract
이 논문은 [Network in Network](https://arxiv.org/pdf/1312.4400.pdf)에서 소개된 Global Average Pooling(이하 GAP)을 재조명하고, 이것이 어떻게 CNN으로 하여금 image-level labels만을 가지고도 훌륭한 localization ability를 갖출 수 있도록 했는지 다룬다.
NiN 논문에서의 GAP는 regularize의 방도로 제안되었지만, 이 논문의 저자들은 GAP가 여러 다른 task들에 사용될 수 있는 *generic localizable deep representation*을 만드는 데에도 기여한다는 것을 밝힌다.
앞으로도 계속 강조되지만, 이 저자들은 자신들이 제안하는 Network가 localization을 위해 학습된 것이 아님에도 불구하고, discriminative image region(discriminative라는 단어는 아마도 image를 discimination할 때, 가장 큰 영향을 주는 부분이라는 것에서 비롯된 듯 함)을 localize하는 능력이 있음을 보여준다.

## 1. Introduction
[Object Detectors Emerge in Deep Scene CNNs](https://arxiv.org/abs/1412.6856)에서 CNN의 convolutional unit들이 location에 대한 정보없이도 object detector로써의 역할을 하고 있음을 보였다. 하지만 이런 정보들은 FC를 통과하면서 유실되게 된다.
다른 한편 NiN과 GoogLeNet의 저자들은 FC를 사용하지 않으므로써 parameter의 수를 줄이는 방법을 제안하였다.
이때, NiN의 저자들은 GAP라는 것을 제안하는데, 이는 FC의 많던 parameter의 수에 비해 상대적으로 적은 수의 parameter를 가지게 된 NiN이 overfitting이 되는 문제를 해결하기 위해서이다.(즉, regularization의 용도로 제안하였다.)

한편 이 논문의 저자들은 이 GAP의 약간의 변형을 가하는 것만으로 final layer까지 앞서 말한 Network의 localization정보를 유지할 수 있음을 알게 되었다. 즉, 아래의 Fig. 1과 같이 classification을 위해 학습된 Network가 classification에 중요한(예를 들면, 양치질의 칫솔) 부분을 localization하고 있음을 알 수 있다.

![intro](../assets/CAM/intro.png)

## 1.1 Related Work
이 논문은 크게 2가지 주제의 논문으로부터 영향을 받았다.

1. Weakly-supervised object localization.
2. Visualizing the internal representation of CNNs.

### Weakly-supervised object localization
이전까지 다양한 제안들이 있었으나, 이러한 논문들에서 저자들은 실제로 localization 능력을 측정한 경우는 없었고, 또한 좋은 결과를 산출했음에도 불구하고 end-to-end로 학습한 모델들이 아니었으며, multiple forward pass를 필요로 하여 real-world datasets에 적용하기 어려웠다.
이 논문의 저자들은 end-to-end로 학습가능하고 single forward pass만을 필요로 하는 모델을 제안한다.

이 논문과 가장 밀접한 연관을 지닌 논문은 [Is object localization for free? - Weakly-supervised learning with convolutional neural network, Oquab *et al*](https://ieeexplore.ieee.org/document/7298668/)이다.
Oquab *et al*은 GAP대신 Global Max Pooling을 사용하였는데, 이 논문의 localization은 해당 object의 boundary 안에 있는 point들을 짚어낼 순 있으나 full extent를 알아낼 수는 없다.
이는 GAP의 loss는 모든 discriminative region을 고려해야 benefits을 얻을 수 있기 때문으로 보인다.

이 논문의 저자들은 이 논문의 가치가 GAP가 accurate discriminative localization에 사용될 수 있음을 보이는 것에 있다고 강조한다.

### Visualizing CNNs
지금까지 CNN의 속성을 더 잘 알기 위해 internal representation을 visualize하려는 시도가 꾸준히 있어왔다.
예를 들어, deconvolution을 이용해 각 unit이 input image의 어느 부위를 보고 있는지 시각화하거나 배경을 detect하기 위해 학습된 Network가 object localization 능력도 있음을 보인다거나 하는 등의 시도들 말이다.
하지만 이런 논문들은 마지막에 들어가는 FC layer부분들을 무시하므로써 완전하지 못한 분석을 하였다. 이 FC를 걷어내므로써, 이 논문의 저자들은 Network의 시작부터 끝까지를 분석할 수 있게 되었다고 주장한다.

추가적으로 이 논문의 저자들은 이 논문에서 제안하는 방식은 image의 정확히 어떤 region이 discrimination에 중요한 역할을 했는지 강조할 수 있다고 한다.

## 2. Class Activation Mapping (CAM)




