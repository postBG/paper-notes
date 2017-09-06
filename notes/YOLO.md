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



