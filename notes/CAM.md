# [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)

## Abstract
이 논문은 [Network in Network](https://arxiv.org/pdf/1312.4400.pdf)에서 소개된 Global Average Pooling(이하 GAP)을 재조명하고, 이것이 어떻게 CNN으로 하여금 image-level labels만을 가지고도 훌륭한 localization ability를 갖출 수 있도록 했는지 다룬다.
NiN 논문에서의 GAP는 regularize의 방도로 제안되었지만, 이 논문의 저자들은 GAP가 여러 다른 task들에 사용될 수 있는 *generic localizable deep representation*을 만드는 데에도 기여한다는 것을 밝힌다.
앞으로도 계속 강조되지만, 이 저자들은 자신들이 제안하는 Network가 localization을 위해 학습된 것이 아님에도 불구하고, discriminative image region(discriminative라는 단어는 아마도 image를 discimination할 때, 가장 큰 영향을 주는 부분이라는 것에서 비롯된 듯 함)을 localize하는 능력이 있음을 보여준다.

## 1. Introduction
[Object Detectors Emerge in Deep Scene CNNs](https://arxiv.org/abs/1412.6856)에서 CNN의 convolutional unit들이 location에 대한 정보없이도 object detector로써의 역할을 하고 있음을 보였다. 하지만 이런 정보들은 FC를 통과하면서 유실되게 된다.
다른 한편 NiN과 GoogLeNet의 저자들은 FC를 사용하지 않으므로써 parameter의 수를 줄이는 방법을 제안하였다.
이때, NiN의 저자들은 GAP라는 것을 제안하는데, 기존의 FC layer가 해석하기 어려우며, 동시에 쉽게 Overfitting되는 문제를 해소하기 위해서였다.(즉, regularizer를 위해 도입)

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
여기서는 CNN의 GAP를 이용해 *class activation mapping(CAM)*을 만드는 방법을 소개한다.
한 카테고리에 대한 CAM은 이 카테고리를 인지하는데 사용되는 discriminative image region을 의미한다.

이 논문에서는 NiN이나 GoogLeNet과 같은 Network를 사용했으며, 이 Network는 output layer전까지는 conv layer로 이뤄져 있고, 그 다음 GAP를 거친 결과를 FC에 넣고 Softmax로 output을 만드는 구조이다.


![CAM](../assets/CAM/network_structure_and_cam.png)

우선 위에서 말한 연구에 사용된 Network에 대해 몇 가지 식을 정리해보도록 하자.
GAP(GAP의 입력으로 들어가는 feature map의 갯수는 classification의 category수와 동일)의 입력으로 들어가기 직전의 마지막 conv layer의 k번째 feature map의 (x, y)에서의 activation 값을 다음과 같이 정의하자.

![act](../assets/CAM/activation_eq.png)

한편 GAP는 각 feature map의 activation의 값들을 전부 더해 평균을 낸 것인데, 이를 수식으로 표현하면 아래와 같다.

![GAP](../assets/CAM/GAP.png)

마지막으로 class c의 값을 나타내는 softmax의 입력은 다음과 같이 표현된다.

![GAP](../assets/CAM/input_to_softmax.png)

한편, 이 식은 다음과 같이 바꿔 쓸 수 있다.

![GAP_](../assets/CAM/softmax_input.png)

여기서 CAM ![CAM_def](../assets/CAM/CAM_def.png)를 다음과 같이 정의한다.

![CAM](../assets/CAM/CAM_eq.png)

그러면, 당연히 다음과 같이 ![S_c_def](../assets/CAM/S_c_def.png)를 다시 쓸 수 있다.

![S_c](../assets/CAM/S_c.png)

그렇기 때문에, ![M_c_xy](../assets/CAM/M_c_xy.png)는 image를 class c라고 예측하게 만드는데 (x, y) 위치의 activation이 얼마나 기여를 했는지 직접적으로 나타낸다.

직관적으로, 우리는 각 unit들이 자신의 receptive field에서의 특정한 visual pattern에 반응한다는 것을 알고 있다. 즉, ![act](../assets/CAM/activation_eq.png)는 map of the presence of visual pattern이라고 볼 수 있다.
그렇다면 CAM은 Fig. 2에서 볼 수 있듯이 특정 위치(= (x, y) 위치의 receptive field 정도로 해석하면 될 듯)에서의  weighted linear sum of the presence of visual patterns로 해석할 수 있다.
그렇기 때문에, CAM을 input image size로 upsampling하므로써 해당 클래스와 가장 큰 관련성을 가진 지역을 Fig. 3과 같이 확인할 수 있다.

![CAM_vis](../assets/CAM/CAM_vis.png)

### Global Average Pooling (GAP) vs Global Max Pooling (GMP)
GAP와 GMP를 사용했을 때의 차이점을 간단히 살펴보자.
저자들은 GAP를 사용하면, loss를 줄이기 위해 관련된 모든 discriminative한 영역을 찾으려 할 것이므로 GMP에 비해 full extent에 더 가까운 영역을 본다고 해석했다.

## Weakly-supervised Object Localization
여기서는 CAM의 localization ability를 측정해보는 과정을 담았다.
포인트는 localization 능력을 학습하기 위해 classification 능력을 크게 훼손하지 않는 것이다.

### 3.1 Setup
앞에서 언급한 것과 같이 여러 종류의 CNN(AlexNet, VGG, GoogLeNet)의 FC를 제거하고, fully-connected softmax layer와 연결된 GAP를 추가한 network를 사용했다.
