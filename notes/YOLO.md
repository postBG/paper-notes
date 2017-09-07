# [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)

**주의**: 같은 주제의 다른 논문을 읽어봐야할 필요성이 있음



기존의 Object-detection 방식은 Object가 존재할 것으로 여겨지는 bounding box들을 제안하는 모델과 그 안의 object를 classify하는 모델들을 pipeline화하여 동작하였기 때문에, 여러가지 metric이 필요하고, 각각의 model을 tuning한 뒤 전체 pipeline도 tuning하는 복잡한 방식으로 훈련 및 테스트 하였다. 그리고 복잡한 pipeline으로 인해 실제 동작은 느린 경우가 많았다.



## 대략적인 동작 방식

저자들은 위의 불편함을 해소하기 위해 Object Detection 문제를 regression problem으로 대치하여 하나의 모델만으로 훈련시키는 방법을  선택하였다.

* 전체 Image를 입력으로 받아 S by S의 grid cell로 쪼갠다.
* 각 cell마다 
  * Grid cell이 Object를 가지는 조건에서 class probability를 계산 (=Pr[Class_i | Object])
  * B개의 bounding boxes와 그 box의 confidence score를 계산 (=Pr[Object] * IOU)
    * confidence score란 제안한 bounding boxes에 object의 존재여부에 대한 confidence와 제안한 box가 얼마나 정확한 곳에 정확한 크기로 존재하는가에 대한 confidence라고 이해하면 됨
* 마지막으로 이전 단계에서 얻은 두가지의 값을 곱하면 class-specific confidence score가 나옴
  * **Pr[Class_i | Object]** * **Pr[Object]** * **IOU** = **Pr[Class_i] * IOU**
  * 결론적으로 YOLO의 net은 Cell마다 bounding box의 cordinate인 (x, y, w, h, confidence)와 class probability를 output으로 가진다.




## Loss function design

**Idea**

* Cell에 object가 없는 경우가 많은데, 이런 경우에 object가 존재하는 cell들의 값들을 압도할 수 있기 때문에, 이에 대한 매커니즘이 필요하다.
  * 두 경우의 가중치를 달리하여 해결
* 또 같은 크기의 deviation이 존재하더라도 bounding box의 크기에 따라 그 중요도 달라지는데, 예를 들면 IOU관점에서 bounding box가 큰 경우에 deviation이 좀 크더라도 괜찮지만, 작은 크기의 box에서는 약간의 deviation도 큰 문제가 될 수 있다. 따라서 둘의 차이를 반영해야 한다.
  * 이를 위해 w와 h의 오차를 계산할때는 squared root를 씌워서 계산
  * 이는 큰 사이즈의 box의 오차를 줄여주는 효과가 존재



## Limitation

YOLO는 각 cell마다 제한된 갯수의 bounding box만을 propose히고 그 box들도 오직 1개의 class만을 가져야하는 강한 spatial contraints를 가지기 때문에, 작은 object들이 무리를 지어 등장하는 경우 예측의 어려움을 겪는다.

또 data와는 다르게 unusual한 비율을 가지는 물체도 예측하는데 어려움을 겪는다.