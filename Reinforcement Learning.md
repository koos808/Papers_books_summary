# 강화학습

## A3C

* 참고 : RLCode와 A3C 쉽고 깊게 이해하기(https://www.youtube.com/watch?v=gINks-YCTBs&t=357s)
* 

* A3C : Asynchronous Advantage Actor-Critic

* 개요
    * 1.샘플 사이의 상관관계를 비동기 업데이트로 해결
    * 2.Replay memory를 사용하지 않음
    * 3.policy gradient 알고리즘 사용가능(Actor-Critic)
    * 4.상대적으로 빠른 학습 속도(여러 에이전트가 환경과 상호작용)
    * 즉, AC = 비동기 + `Actor-Critic`
      * `Actor-Critic`이라는 Agent와 환경을 여러개 만들어서 비동기적으로 업데이트 해나가는 것이 A3C이다.

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
