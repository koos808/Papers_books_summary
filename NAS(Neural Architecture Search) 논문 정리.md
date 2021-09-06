# 0. Nas Map

<U>NAS 공부를 위한 논문 순서와 참고 블로그 정리</U>

* **개인적인 공부를 위한 자료입니다.**
* 참고 사이트
  * http://khanrc.github.io/nas-1-intro.html

* 논문 읽는 순서 및 참고 사이트 정리
  1. NASRL : Neural Architecture Search with Reinforcement Learning, Barret Zoph et al., 2016
  2. NASNet : NasNet, Learning Transferable Architectures for Scalabel Image Recognition
  3. ENAS : Efficient Neural Architecture Search via Parameter Sharing
    * https://jayhey.github.io/deep%20learning/2018/03/15/ENAS/
  4. MnasNet: Platform-Aware Neural Architecture Search for Mobile


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

* 3.Methods - **개요**
  * 가장 먼저 `recurrent network`를 사용해서 convolutional 아키텍처를 생성하는 방법을 설명한다. 그 다음, 샘플링된 아키텍처의 `expected accuracy를 maximize`하기 위해 `policy gradient` 방법으로 recurrent network를 학습하는 방법을 설명한다.
  * model complexity(모델 복잡성)을 증가시키기 위해 `skip connections`을 형성한다. 또한, 훈련 속도를 높이기 위해 parameter server 접근 방식을 사용하는 것과 같은 핵심적인 접근 방식의 `몇 가지 개선 사항`을 제시한다.

* **3.1 GENERATE MODEL DESCRIPTIONS WITH A CONTROLLER RECURRENT NEURAL NETWORK**
  * 컨볼루션 레이어만 있는 피드포워드 신경망을 예측하고 싶다고 가정하고 controller를 사용하여 hyperparameters를 토큰 시퀀스로 생성할 수 있다.
  * <image src="image/NAS_1_Figure2.png" style = "width:400px">
  * Figure 2는 controller recurrent neural network가 간단한 convolutional network를 샘플링하는 방법이다. 한 레이어에 대한 필터 높이, 필터 너비, 보폭 높이, 보폭 및 필터 수를 예측하고 반복한다. 모든 예측은 softmax 분류기에 의해 수행된 다음 입력으로 다음 단계에 제공된다.
  * 특이사항
    * In our experiments, the process of generating an architecture stops if the number of layers exceeds a certain value. => <U>레이어 수가 특정 값을 초과하면 아키텍처 생성 프로세스가 중지된다.</U>
  * 레이어 수의 특정 값은 훈련이 진행됨에 따라 증가하는 스케쥴을 따른다. 컨트롤러 RNN이 아키텍처 생성을 마치면 이 아키텍처로 신경망이 구축되고 훈련된다. Convergence 시에 보류된 validation set에 대한 네트워크의 accuracy가 기록된다.
  * 그런 다음 컨트롤러 RNN의 매개변수 $θ_c$는 제안된 아키텍처의 expected validation accuracy를 최대화하기 위해 Optimized된다.
  
* **3.2 TRAINING WITH REINFORCE**
  * 이번 섹션에서는 컨트롤러 RNN이 시간이 지남에 따라 더 나은 아키텍처를 생성하도록 매개변수 $θ_c$를 업데이트하는 데 사용하는 policy gradient method를 설명한다.
  * convergence 시 child network는 보류된 데이터세트에서 accuracy "R"을 얻게 된다. 이 정확도 "R"을 **reward signal**로 사용하고 강화학습을 사용하여 controller를 훈련한다.
  * 구체적으로, 최적의 아키텍처를 찾기 위해 controller에 $J(θ_c)$로 표시되는 exptected reward를 maximize하도록 한다.
  * => <image src="image/NAS_1_EQ1.png" style = "width:300px">
    * controller가 예측하는 토큰 list는 child network의 아키텍처를 설계하기 위한 $a_{1:T}$ **actions** list와 같다. 
<br><br>
  * **reward signal R**은 non-differenctiable(미분 불가능)하므로 $θ_c$를 반복적으로 업데이트하기 위해 policy gradient 방법을 사용해야 한다. 여기에는 REINFORCE rule(Williams,1992)을 사용한다.
  * => <image src="image/NAS_1_EQ2.png" style = "width:500px">
  * 위 EQ(2)의 Quantity는 아래와 같다.
  * => <image src="image/NAS_1_EQ3.png" style = "width:400px">
    * m : controller가 한 batch에서 샘플링하는 서로 다른 아키텍처의 수
    * T : controller가 neural network 아키텍처를 설계하기 위해 예측해야 하는 hyperparameters의 수
    * $R_k$ : k 번째 neural network 아키텍처가 training 데이터 세트에 대해 학습된 후 달성하는 검증 정확도
  * 위의 업데이트는 gradient(기울기)에 대한 unbiased estimate(편향되지 않은 추정치)이지만 분산이 매우 높다. 이 추정치의 분산을 줄이기 위해 baseline function을 사용한다.
    * => <image src="image/NAS_1_EQ4.png" style = "width:400px">
    * baseline function b가 `현재 action`에 depend하지 않는 한 이것은 여전히 unbiased gradient estimate이다. 이 논문에서는 baseline b를 `이전 아키텍처 정확도의 exponential moving average(지수 이동 평균)`으로 사용했다.

* **Accelerate Training with Parallelism and Asynchronous Updates**
  * 말 그대로 병렬화와 비동기식(asynchronous) 업데이트로 학습 과속화에 대한 내용이다.
  * Neural Architecture Search에서 컨트롤러 매개변수 $θ_c$에 대한 각 gradient 업데이트는 하나의 child network를 convergence하도록 학습하는 것에 해당한다.
  * <u>모든 child network를 훈련하는데 오래 걸릴 수 있고 controller의 학습 프로세스 속도를 높이기 위해 분산 학습(distributed training)과 비동기식 매개변수 업데이트(asynchronous parameter updates)를 사용한다.</u>(Dean et al,2012)

  * => <image src="image/NAS_1_Figure3.png" style = "width:700px">
    * Figure 3: Distributed training for Neural Architecture Search.
    * shards S : K controller replicas(복제본)에 대한 shared parameters(공유 파라메터)를 저장하는 공간
    * S shards의 parameter server가 있는 parameter-server 체계를 사용한다. (shard는 일종의 파티션과 같은 의미, 데이터 조각 단위라고 이해하면 편함)
    * 각 컨트롤러 replica는 병렬로 훈련된 m개의 서로 다른 하위 아키텍처를 샘플링합니다.
    * 그런 다음 컨트롤러는 convergence 시 "m" 아키텍처의 mini-batch 결과에 따라 gradient를 수집하고 모든 컨트롤러 복제본에서 **weights**를 업데이트하기 위해 parameter server로 보낸다. 이 논문에서 각 child network의 수렴은 학습이 특정 Epoch 수를 초과할 때 도달한다.
    * <u>Figure 3 요약 : S parameter server 세트를 사용하여 parameter를 저장하고 K controller replicas로 보낸다. 그런 다음 각 컨트롤러 복제본은 m개의 아키텍처를 샘플링하고 여러 하위 모델을 병렬로 실행한다. 각 하위 모델의 정확도가 기록되어 $θ_c$에 대한 gradient를 계산한 다음 매개변수 서버로 다시 전송된다.</u>


* **3.3 Increase Architecture Complexity With SKIP CONNECTIONS and other layer types**
  * `개요` : section 3.1에서 search space에는 GoogleNet이나 Residual Net과 같은 skip-connections이나 branching layers(분기 레이어)가 없다. 따라서 controller가 skip-connectionrhk branching layers를 제안하여 search space를 넓힐 수 있는 방법을 제안했다.
  * controller가 이러한 연결을 예측할 수 있도록 하기 위해 attention mechanism을 기반으로 구축된 set-selection type attention을 사용한다.
  * 레이어 N에서 연결해야 하는 이전 레이어를 나타내기 위해 N − 1개의 content-based sigmoid가 있는 anchor(앵커) point를 추가한다. 각 시그모이드는 컨트롤러의 현재 hidden state와 이전 N − 1개의 앵커 포인트의 이전 hiddens tates의 function이다.
  * => <image src="image/NAS_1_EQ5.png" style = "width:680px">
     * $h_j$ : j 번째 레이어에 대한 anchor point에서 controller의 hidden state를 나타낸다. 그리고 j의 범위는 0에서 N-1까지이다.
     * 그런 다음 이 시그모이드에서 샘플링하여 현재 레이어에 대한 입력으로 사용할 이전 레이어를 결정한다. 행렬 $W_{prev}, W_{curr}, v$ 는 trainable parameter이다. <br><br>
  * <image src="image/NAS_1_Figure4.png" style = "width:600px"> <br>
    * Figure 4는 컨트롤러가 현재 layer에 대한 input으로 원하는 layer를 결정하기 위해 skip-connection을 사용하는 방법을 보여준다.

  * **skip-connections 문제 발생 가능성** : Skip connections은 한 layer가 다른 layer와 호환되지 않거나 한 layer에 input 또는 output이 없을 수 있는 "compilation failures(컴파일 실패)"를 유발할 수 있다.
    * 해결 방법 1) 첫째, 레이어가 입력 레이어에 연결되어 있지 않으면 이미지가 입력 레이어로 사용된다. 
    * 해결 방법 2) 둘째, 최종 layer에서 연결되지 않은 모든 layer output을 가져와 연결한 다음 이 최종 hidden state를 classifier로 보낸다.
    * 해결 방법 3) 셋째, 연결할 input layer의 크기가 다른 경우 연결된 레이어의 크기가 같도록 작은 레이어를 0으로 채운다.
  * 섹션 3.1에서 learning rate를 예측하지 않고 아키텍처가 convolutional layer로만 구성되어 있다고 가정했다. 하지만, 학습률을 예측 중 하나로 추가할 수 있으며, 더 나아가 pooling, local contrast normalization, batchnorm 등의 예측을 추가하는 것도 가능하다. 더 많은 유형의 layer를 추가할 수 있으려면 controller rnn에 추가 단계를 추가하여 layer 유형을 예측한 다음 이와 관련된 다른 하이퍼파라미터를 추가해야 한다.

* **3.4 GENERATE RECURRENT CELL ARCHITECTURES**
  * 모든 time step t에서 컨트롤러는 $x_t$ 및 $h_{t−1}$을 입력으로 사용하는 $h_t$에 대한 functional form을 찾아야 하는데, 가장 간단한 방법은 basic recurrent cell 공식인 $h_t = tanh(W_1 ∗ x_t +W_2∗h_{t−1})$ 이다.
  * 기본 RNN 및 LSTM 셀에 대한 계산은 $x_t$를 취하는 단계의 트리로 일반화할 수 있다. 컨트롤러 RNN은 두 개의 입력을 병합하고 하나의 출력을 생성하기 위해 조합 방법(덧셈, 요소별 곱셈 등)과 활성화 함수(tanh, sigmoid 등)로 트리의 각 노드에 레이블을 지정해야 한다.
  * 그런 다음 두 개의 출력이 트리의 다음 노드에 대한 입력으로 제공된다. 컨트롤러 RNN이 이러한 방법과 functions를 선택할 수 있도록 하기 위해 컨트롤러 RNN이 각 노드를 하나씩 visit하고 필요한 하이퍼파라미터에 레이블을 지정할 수 있도록 트리의 노드를 순서대로 indexing한다.
  * LSTM 셀의 구성에서 영감을 받아서 `Memory state`를 나타내기 위해 cell variables $c_{t-1}$과 $c_t$도 필요하다. 이러한 변수를 통합하려면 컨트롤러 RNN이 이 두 변수를 연결할 트리의 노드를 예측해야 한다. 이러한 예측은 컨트롤러 RNN의 마지막 두 블록에서 수행한다.<br><br>
  * <image src="image/NAS_1_Figure5.png" style = "width:700px"> <br>
    * 왼쪽: 컨트롤러가 예측할 계산 단계를 정의하는 트리.
    * 가운데: 트리의 각 계산 단계에 대해 컨트롤러가 수행한 예측 집합의 예.
    * 오른쪽: 컨트롤러의 예제 예측에서 구성된 순환 셀의 계산 그래프.<br><br>
  * 요약 : 이번에는, LSTM과 유사한 RNN의 기본 연산 단위인 cell을 생성해봤다. RNN의 기본 cell은 input, output 뿐만 아니라 memory state로 구성되는데, 이 모델은 tree 형태로 일반화할 수 있다. 위 그림은 왼쪽의 가장 간단한 형태의 tree 모델을 Neural Architecture Search를 거쳐 오른쪽의 기본적 unit들의 조합으로 만들어낸 예제이다. 즉, LSTM이나 GRU와 같은 RNN의 cell을 이렇게 자동으로 만들어 내는 것이 가능하다.

* 4.Experiment 간단하게.
  * Task : CIFAR-10(image classification), Penn Treebank(language modeling)
  * 목적 : CIFAR-10에서 목표는 좋은 컨볼루션 아키텍처를 찾는 반면 Penn Treebank에서 목표는 좋은 recurrent cell을 찾는 것이다.
  * 분산 학습을 위해 800개의 GPU를 사용 (S:20, K:100, m:8).
  * Search space : 비선형 함수로 Relu를 사용하며 각 레이어마다 batch normalization과 skip connection이 존재한다. 모든 convolutional 레이어에 대해, RNN controller는 filter의 height[1, 3, 5, 7]과 width[1, 3, 5, 7], 그리고 필터의 수[24, 36, 48, 64]를 선택한다. stride에 대해서는 두 가지의 실험을 하는데, 한 가지는 1로 고정시킨 것이며, 다른 건 controller가 [1, 2, 3]의 값 중 하나를 선택하게 하는 것이다.



2.NasNet: Learning Transferable Architectures for Scalabel Image Recognition
===

* 간단 요약
  * 전체 네트워크를 search 하는 것이 아닌, 셀 구조를 search해서 stack하는 방식으로 적용.
  * `Normal cell`과 `Reduction cell`(feature map size 줄이기) 두개를 각각 찾음.
  *  search space 미리 정의 -> filter size와 stride size 등을 자주 사용하는 것들로 미리 구성하며 skip-connection 또한 2개로 고정
  *  RNN은 `2`(normal cell, reduction cell) X `5`(논문에서 conv block 5개로 사용)
     *  RNN은 2x5B의 softmax prediction을 하며, B는 cell을 구성하는 convolution block의 개수 의미함.

3.ENAS : Efficient Neural Architecture Search via Parameter Sharing
===

* 간단 요약
  * 이 논문의 가장 큰 Contribution은 `child model들이 weight를 공유하게 만들어서 밑바닥부터 학습하지 않게 하는 것`임. 즉, child model 간 파라미터 공유 뿐만 아니라 좋은 성능까지 유도해서 계산 복잡도와 퍼포먼스 두 개를 동시에 달성한 것.
  * `Parameter sharing` 아이디어 제안
    * 위치와 인풋, 연산이 동일하면 서로 parameter를 공유함.
  * 이전 논문들은 RNN controller가 mini-batch 단위로 네트워크 구조를 생성하고 그 구조들을 전부 학습한 뒤에 업데이트했는데, ENAS에서는 RNN controller가 생성한 네트워크 구조들을 1-step씩만 학습시켜 성능 비교 및 제일 좋은 구조 하나만 학습함.
  * search space를 엄청 줄임.

4.MnasNet: Platform-Aware Neural Architecture Search for Mobile
===

* 간단 요약
  * 모바일 플랫폼에서 사용 가능한 NAS를 찾는 것이 목표이기 때문에, 기존의 validation accuracy를 objective function으로 최적화하는 것이 아닌, 정확도를 조금 낮추더라도 가벼운 네트워크를 만드는 것에 초점을 맞춘 논문이다.
  * validation accuracy와 latency(inference spped)를 Reward로 사용하여 최적화함.
  * latency constraint를 soft하게 걸어줌으로써 Pareto optimal을 얻는다.














