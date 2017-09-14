# Generative Adversarial Nets

#### 요약

GAN은 두 가지의 모델(generative model G, discriminative model D)이 서로 경쟁하며 학습하는 방식이다. 

G는 random noise로부터 real data와 비슷한 fake data를 generate해 D를 속이기 위해 학습하고, D는 G에게 속지 않고 real과 fake를 구별하기 위해 학습한다.



![GAN objective function](../assets/GAN/obj.PNG)

 D(x)를 x가 real일 확률이라고 한다면,

* D의 입장에서는 위의 식을 maximize하기 위해 input으로 fake(=G(z))가 들어왔을 때는 0을, real이 들어왔을 때는 1을 내보내도록 학습하게 된다.
* G의 입장에서는 위의 식을 minimize하기 위해 1 - D(G(z))가 minimize되야하므로 D(G(z))가 1이 되도록 G(z)가 P_data를 따르도록 학습해야 한다.



#### key points

* G가 학습되는 동안 D는 optimal solution에 가까운 상태여야 한다. 이는 D의 overfitting을 방지하기 위함인데, 저자들은 G를 1번의 iteration을 학습하는 동안 D는 k iteration을 학습하는 방법을 사용하였다.

* 충분한 capacity와 training time이 있다면, GAN의 G가 학습하는 분포 p_g가 p_data로 수렴함을 증명할 수 있다.

* 실제로 학습시킬 때, 학습 초기에 D가 G가 학습이 덜 된 상태에서 만들어낸 fake를 보고 높은 confidence로 reject를 하면 G의 학습이 어려워질 수 있어(log(1- D(G(z)))가 saturate) log(D(G(z)))를 maximize하도록 학습시켰다.

* G는 D로부터 흘러온 gradients만으로 학습하므로, input이 G의 parameter에 그대로 copy되는 문제를 겪지 않는다.

* *Semi-supervised learning*처럼 제한된 labeled data만 사용해야하는 경우에 도움을 줄 수 있다.

* 다른 generative model과는 달리 degenerative distribution이나 sharp한 distribution들도 학습할 수 있다.

  ​