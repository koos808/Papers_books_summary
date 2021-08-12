1.Neural Architecture Search with Reinforcement Learning, Barret Zoph et al., 2016
===

* 0.Abstract
  * `모델 측면`
    * 신경망의 model descriptions을 생성하기 위해 Recurrnet network를 사용한다.
    * **Validation set에서 생성된 architectures의 expected accuracy를 maximaize하기 위해서 reinforcement learning으로 RNN을 학습시킨다.**
    * Penn Treebank 데이터 세트에서 저자들의 모델은 널리 사용되는 LSTM cell 및 다른 SOTA를 능가하는 새로운 recurrent cell을 구성할 수 있다.
  * `실험 측면`
    * CIFAR-10 데이터 셋트에서 저자들이 design한 novel network architecture는 이전에 있던 best human-invented architecture에 필적할 만한 성능을 갖는다.
    * Penn Treebank에서 state-of-the-art perplexity을 달성했다.

* 1.Introduction
  * This paper presents `Neural Architecture Search`, a gradient-based method for finding good architectures (see Figure 1).
  * <image src="image/NAS_1_Figure1.png" style = "width:400px">
  * 이 논문의 Work는 neural network의 structure와 connectivity가 일반적으로 variable-length string(가변 길이 문자열)로 지정될 수 있다는 observation을 기반으로 한다. (Our work is based on the observation that the structure and connectivity of a neural network can be typically specified by a variable-length string.)
  * <U>실제 데이터에 대해 `"child network"`라는 string으로 지정된 network를 학습하면 validation set의 정확도가 나온다. 이 정확도를 **reward signal**로 사용하여 policy gradient를 계산하고 RNN으로 구성된 controller를 업데이트 할 수 있다.</U>
  * 결과적으로 다음 iteration에서 controller는 높은 정확도를 받는 아키텍처에게 더 높은 probabilities를 부여하게 된다. 즉, controller는 시간이 지날수록 search를 개선하는 방법을 학습하게 되는 것이다.

* 2.Related Work
  * `Hyperparameter optimization`
    * 이러한 방법의 성공에도 불구하고 고정 길이 공간(fixed-length space)에서만 모델을 검색한다는 점에서 여전히 제한적이다. 즉, 네트워크의 구조와 연결성을 지정하는 가변 길이 구성(variable-length configuration)을 생성하도록 요청하기가 어렵다.
    * 이러한 방법은 좋은 초기 모델이 제공되는 경우 더 잘 작동하는 경우가 많다.
  * `Bayesian optimization`
    * Bayesian optimization은 고정 길이가 아닌 아키텍처를 검색할 수 있는 최적화 방법이지만, 이 논문에서 제안한 방법보다 유연성과 일반성이 떨어진다.
  * `evolution algorithms`
    * 새로운 모델을 구성하는 데 훨씬 유연하지만, 일반적으로 대규모에서는 덜 실용적이다. search-based methods가 느리거나 제대로 작동하려면 많은 경험적인(heuristics) 방법이 필요하다는 한계가 있다.
  * `해당 논문의 Neural Architecture Search`
    * Neural Architecture Search의 **컨트롤러(controller)**는 **자동 회귀(auto-regressive)** 이다. 즉, 이전 예측에 따라 하이퍼파라미터를 한 번에 하나씩 예측한다.
    * 이 아이디어는 end-to-end sequence to sequence learning의 디코더에서 차용되었다(Sutskever et al., 2014).
    * sequence to sequence learning과 달리 이 논문의 방법은 child network의 정확도인 미분할 수 없는(non-differentiable) 메트릭을 최적화한다. 따라서 Neural Machine Translation의 BLEU 최적화 작업과 유사하다.

* 3.Methods

* 개요
  * 가장 먼저 `recurrent network`를 사용해서 convolutional 아키텍처를 생성하는 방법을 설명한다. 그 다음, 샘플링된 아키텍처의 `expected accuracy를 maximize`하기 위해 `policy gradient` 방법으로 recurrent network를 학습하는 방법을 설명한다.
  * model complexity(모델 복잡성)을 증가시키기 위해 `skip connections`을 형성한다. 또한, 훈련 속도를 높이기 위해 parameter server 접근 방식을 사용하는 것과 같은 핵심적인 접근 방식의 `몇 가지 개선 사항`을 제시한다.

