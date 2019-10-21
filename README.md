Papers & Books Summary Repository
===========

#### Explanation : A summary of statistical papers, books, etc.

*※ Statistics*
===
* 확률(Probability)
    * 1.확률이란?
        * 어떤 시행에서 사건 A가 일어날 가능성을 수로 나타낸 것을 사건 A가 일어날 확률이라 하고, 이것을 기호로 P(A)로 나타낸다. 이때, P는 probability(확률)의 첫글자이다.
    * 2.확률의 종류
        * 1. 수학적 확률
            * 
            ``` 
            수학적 확률은 확률의 고전적 정의이다.
            어떤 시행에서 각각의 경우가 일어날 가능성이 같다고 할 때, 
            일어날 수 있는 모든 경우의 수를 s, 어떤 사건 A가 일어날 경우의 수를 a라고 하면
            사건 A가 일어날 확률 P(A)는 a/s이다. 이와 같이 정의된 확률을 수학적 확률이라 한다.
            ```
        * 2. 통계적 확률(경험적 확률)
            * 
            ```
            같은 시행을 n번 반복했을 때의 사건 A가 일어난 횟수를 r이라고 할 때,
            n을 한없이 크게 함에 따라 상대도수 r/n이 일정한 값 p에 가까워지면
            이 p를 사건 A의 통계적 확률이라 한다.
            ```
        * 3. 기하학적 확률
    * 3.확률의 공리적 정의
        * ① 표본 공간 S에서 임의의 사건 A에 대하여 0≤P(A)≤1
        * ② 전사건 S에 대하여 P(S)=1(반드시 일어날 때, 확률은 1)
        * ③ mutually exclusive event(배반사건)의 확률의 합들은 각각의 확률의 합과 같다.

* 신뢰구간(Confidence Interval)
    * **신뢰구간을 구하는 이유**는 모평균의 신뢰성을 가늠하기 위해서이다.
        * 모평균은 왜 구하는 것일까? 우리가 어떤 자료를 파악하고자 할 때는 그 자료의 평균이나 분산 등의 값들을 먼저 구한다. 평균을 알면 자료의 대표적인 값을 알 수 있고 분산을 알면 자료가 평균으로부터 얼마나 떨어져 있는지를 파악할 수 있기 때문이다. **but 모평균과 모분산을 직접 계산한다는 것은 일반적으로 거의 불가능하다. -> 모집단의 원소는 일반적으로 매우 크기 때문!!**
        * 따라서 조사하고자 하는 어떤 거대한 모집단이 존재한다면, 표본을 추출하여 모평균 혹은 모분산을 **추정**하는 것이 통계학의 가장 기본적인 방법이다. 이렇게 추출된 표본으로부터 구한 표본평균 및 표본분산을 **모평균과 모분산의 추정치**로 사용한다.
        * **하지만** 추정치를 100% 신뢰할 수 없으므로 추정치들의 모평균(모분산)에 대한 신뢰구간을 구함으로써 그 신뢰성을 어느 정도 측정할 수 있게 만들었다.
    * **신뢰구간의 의미**
        * 추출된 표본이 정해진 개념이 아니듯이 신뢰구간 또한 마찬가지로 *명확히 정해지는 개념이 아니다.*
        * **같은 방법으로 100번 표본을 추출했을 때, 함께 계산되는 100개의 신뢰구간 중 모평균을 포함한 신뢰구간들의 개수는 95개 정도 된다.** 라는 의미다.

        * TIP : '모평균을 포함할 확률이 95%가 되는 구간'은 틀린 의미
    * 참고 : http://blog.naver.com/PostView.nhn?blogId=vnf3751&logNo=220823007712

* P-value
    * 정의 : p-value는, 귀무가설(null hypothesis, H0)이 맞다는 전제 하에, 통계값(statistics)1이 실제로 관측된 값 이상일 확률을 의미한다. 일반적으로 p-value는 어떤 가설을 전제로, 그 가설이 맞는다는 가정 하에, 내가 현재 구한 통계값이 얼마나 자주 나올 것인가, 를 의미한다고 할 수 있다.
    * p-value의 필요 이유 : p-value는 가설검정이라는 것이 전체 데이터를 갖고 하는 것이 아닌 sampling 된 데이터를 갖고 하는 것이기 때문에 필요하게 된다.