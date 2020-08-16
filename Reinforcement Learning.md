# 강화학습

## A3C

* 참고1 : RLCode와 A3C 쉽고 깊게 이해하기(https://www.youtube.com/watch?v=gINks-YCTBs&t=357s)
* 참고2 : [쉽게읽는 강화학습 논문 4화] A3C 논문 리뷰(https://www.youtube.com/watch?v=lfaaGQcQ2FE)

* A3C : Asynchronous Advantage Actor-Critic

* 전체를 보여주는 이미지
  * <image src="image/A3C.jpg" style="width:500px">

* 개요
    * 1.샘플 사이의 상관관계를 비동기 업데이트로 해결
    * 2.Replay memory를 사용하지 않음
    * 3.policy gradient 알고리즘 사용가능(Actor-Critic)
    * 4.상대적으로 빠른 학습 속도(여러 에이전트가 환경과 상호작용)
    * 즉, AC = 비동기 + `Actor-Critic`
      * `Actor-Critic`이라는 Agent와 환경을 여러개 만들어서 비동기적으로 업데이트 해나가는 것이 A3C이다.

* Contribution
  * 1.다양한 RL 알고리즘의 **"scale-up"** 방법을 제시!
    * Off/on policy, Value/Policy based ... 모두에서 stable.
    * Experience Replay 대신 병렬적 actor가 decorrelation 가능케 해준다. => RL을 Supervised Learning처럼 학습하기 위해 Experience Replay, Target Network 등을 사용하고 iid 가정이 존재한다. 하지만 위와 같은 iid 가정이 필요없는 방법을 사용함으로써 좀 더 자유로워짐.
    * GPU 대신 CPU Thread를 사용.
    * 심지어 Super Linear... => Actor-learner thread가 1개 있을때보다 16개 있을 때 학습속도가 16배가 되어야 하는데 이것이 Linear하다는 뜻. Super Linear는 학습속도가 1개 썼을때 보다 16배보다 더 빨라진다는 뜻!
  * 2.SOTA 갱신! (State-Of-The-Art)

* `Actor-Critic`은 무엇일까?
    * Actor-Critic = REINFORCE + 실시간 학습 ::: REINFORCE라는 알고리즘을 online으로 학습하는 것.
    * REINFORCE : Policy gradient를 몬테카를로 방법으로 업데이트하는 것
    * `Critic`
      * Neural Network를 의미하며, 업데이트의 방향과 크기를 나타낸다.
      * 큐함수(Q-function)를 나타냄. $Q(S_t,A_t)*(-\sum y_ilogp_i )$ 계산해서 Gradient descent를 함. 여기서 $Q(S_t,A_t)$가 크리틱이다.

* 여러 개의 Agent
  * 여러개의 Agent들을 만들고(보통 16개) 각각 상태와 Action이 있을 때의 $Q(S_{t_1},A_{t_1})*(-\sum y_ilogp_i )$ 를 계산한다. 그리고 Gradient를 계산한다.
  * 이후, Global network를 업데이트 한다.
  * agent n이 비동기적으로 Global network를 업데이트한다.

* Policy gradient
  * 1.정책을 업데이트 하는 기준 : `목표함수`
  * 2.목표함수에 따라 정책을 업데이트하는 방법 : `Gradient ascent`
    * 1) 가치 기반 강화학습(Value-based)
      * 매 스텝마다 에이전트가 행동을 선택하는 기준 -> 가치함수(Value function)
    * 2) 정책 기반 강화학습(Policy-based)
      * 정책을 업데이트할 때마다 어떤 방향으로 업데이트할 지에 대한 기준
        * -> 목표함수(Objective function) or $J(\theta)$

* 목표함수
  * 에이전트가 정책 $\pi_\theta$ 에 따라서 가게 되는 "경로,궤적(trajectory)"을 생각해보자!
  * "경로,궤적(trajectory)" = 에이전트와 환경이 상호작용한 흔적
    * "경로,궤적(trajectory)"는 에피소드(Episode)나 Rollout이라고 부르기도 한다.
    * initial state부터 terminal state까지 에이전트가 거친 (상태, 행동, 보상)의 sequence를 의미한다. 
    * $$\tau = (s_0, a_0, s_1, a_1, ... , s_T).$$
    * $J(\theta)$ = trajectory동안 받을 것이라고 기대하는 Reward의 합(trajectory가 매번 달라지므로 기대값을 사용함)
    * $J(\theta)=E[\sum_{t=0}^{T-1}r_{t+1}|\pi_\theta]=E[r_1+r_2+r_3 + ... +r_T|\pi_\theta]$
    * $J(\theta)$는 높은게 좋은 것이니까 Gradient Descent가 아닌 Gradient Ascent를 한다.(부호 +)

* Gradient Ascent
  * $J(\theta)$를 기준으로 어떻게 $\theta$(정책신경망)을 업데이트할 것인가?
  * ->$\theta$에 대한 $J(\theta)$의 경사를 따라 올라가다(Gradient Ascent)
  * $\theta'$ = $\theta'+\alpha\nabla_\theta J(\theta)$
  * $\nabla_\theta J(\theta)$ = Policy Gradient
  * 즉, 목적함수 $J(\theta)$를 미분한 것을 Policy Gradient라고 한다.

  * ![a](./image/Policy_gradient.jpg)

* REINFORCE
  * Discounted future reward를 Retun G($G_t$)라고 한다.
  * REINFORCE 알고리즘 업데이트 식
    * $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta) = \theta +\alpha\sum_{t=0}^{T-1}\nabla_\theta log\pi_\theta(a_t|s_t)G_t$
  * 절차
    * 1.한 에피소드를 현재 정책에 따라 실행
    * 2.Trajectory를 기록
    * 3.에피소드가 끝난 뒤 $G_t$ 계산
    * 4.Policy gradient를 계산해서 정책 업데이트
    * 5.(1~4) 반복 : 에피소드마다 업데이트하고 반복하므로 몬테카를로 Policy gradient라 부르며 이를 REINFORCE 알고리즘이라고 한다.
  * `REINFORCE 알고리즘의 문제점`
    * Variance가 높다.  ::: 에피소드가 길어질수록 다양한 Trajectory가 된다.
    * 에피소드마다 업데이트가 가능하다(on-line이 아니다.) -> 에이전트와 환경이 상호작용하는 동안에 업데이트를 할 수 없으므로 online이 아닌 offline으로 학습이 된다.

* REINFORCE 알고리즘을 보완한 Actor-Critic
  * 몬테카를로 -> `TD(Temporal-Difference)` 
    * 몬테카를로는 끝날때까지 진행하고 업데이트 하지만, TD는 한번만 진행해보고 업데이트하는 것이다.
  * REINFORCE -> `Actor-Critic`
    * 위의 몬테카를로에서 TD로 바꾸는 방법과 비슷한 뉘앙스이다.
    * 즉, REINFORCE 알고리즘은 에피소드가 끝나야 업데이트가 가능했지만, Actor-Critic은 매 스텝이 끝날 때마다 업데이트가 가능하도록 고안한 것이다.

* `Actor-Critic`
  * $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta) = \theta +\alpha\sum_{t=0}^{T-1}\nabla_\theta log\pi_\theta(a_t|s_t)G_t$ 의 $G_t$를 $Q_{\pi\theta}(s_t,a_t)$로 바꿀 수 있다.
  * 만약 $Q_{\pi\theta}(s_t,a_t)$를 알 수 있다면 매 time-step마다 업데이트하는 것이 가능해진다!!!
  * $Q_{\pi\theta}(s_t,a_t) \sim Q_w(s_t,a_t)$ ::: 즉, 두 개의 네트워크(Q-function과 Policy)를 사용한다는 뜻. 이 부분(Q)이 **Critic**이 된다.
  * 내가 지금 하는 행동에 대해서 Q가 좋은지 안좋은지를 판단해 주는 것이 **Critic**이다.

* `Actor-Critic` - Advantage 함수
  * `Advantage 함수` = 큐함수 - 베이스라인 -> Variance를 낮춰준다.
  * Q-function : 특정 상태, 특정 행동에 따른 값
  * Value-function : 특정 상태, 전반적 행동에 따른 값 -> 베이스라인

* 간단 요약
  * `Actor`
    * 1) 정책(Policy)을 근사 : $\theta$
    * 2) $\nabla_\theta log\pi_\theta (a_t|s_t)(r_{t+1} + \gamma V_v(s_{t+1})-V_v(s_t))$로 업데이트
  * `Critic`
    * 가치함수(Value function)을 근사 : v
    * $(r_{t+1} + \gamma V_v(s_{t+1})-V_v(s_t))^2$의 loss function으로 업데이트

* `A3C`
  * Actor-Critic과 다른점
    * Actor를 업데이트하는 과정에서 아래 2가지 부분이 다르다.
    * 1.Multi-step loss function
      * 1-step => multi-step(bias가 줄일 수 있으며 효율적인 시간활용을 할 수 있다.)
      * ex) 20 step : 20 step마다 20개의 loss function을 더한 것으로 업데이트
    * 2.Entropy loss function
      * 20개의 cross entropy : exploitation