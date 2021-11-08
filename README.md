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
            ``` 
            수학적 확률은 확률의 고전적 정의이다.
            어떤 시행에서 각각의 경우가 일어날 가능성이 같다고 할 때, 
            일어날 수 있는 모든 경우의 수를 s, 어떤 사건 A가 일어날 경우의 수를 a라고 하면
            사건 A가 일어날 확률 P(A)는 a/s이다. 이와 같이 정의된 확률을 수학적 확률이라 한다.
            ```
        * 2. 통계적 확률(경험적 확률)
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
    * 정의 : p-value는, 귀무가설(null hypothesis, H0)이 맞다는 전제 하에, 관측된 통계값 혹은 그 값보다 큰 값이 나올 확률이다. 일반적으로 p-value는 어떤 가설을 전제로, 그 가설이 맞는다는 가정 하에, 내가 현재 구한 통계값이 얼마나 자주 나올 것인가를 의미한다고 할 수 있다.
    * p-value의 필요 이유 : p-value는 가설검정이라는 것이 전체 데이터를 갖고 하는 것이 아닌 sampling 된 데이터를 갖고 하는 것이기 때문에 필요하게 된다.
    * 정리를 하면, 가설검증이라는 것은 전체 데이터의 일부만을 추출하여 평균을 내고, 그 평균이 전체 데이터의 평균을 잘 반영한다는 가정 하에 전체 데이터의 평균을 구하는 작업인데, 아무리 무작위 추출을 잘 한다 하더라도 추출된 데이터의 평균은 전체 데이터의 평균에서 멀어질 수 있게 된다. 따라서, **내가 추출한 이 데이터의 평균이 원래의 전체 데이터의 평균과 얼마나 다른 값인지를 알 수 있는 방법이 필요하게 된다.** 이와 같은 문제 때문에 나온 값이 **p-value** 이다.
        * 우리는 평균이 100 이라는 가정 하에서는 sampling 된 데이터의 평균이 100 근처에 있을 것이라는 생각을 하게 되고, 따라서, 역으로, sampling 된 데이터의 평균이 100에서 멀면 멀수록 모분포의 평균이 100 이 아닐지도 모른다는 생각을 하게 된다.
    * 핵심 : **"모분포의 평균이 100 이다"라는 귀무가설이 참이라는 가정 하에서, 100개의 데이터를 sampling할 때 이론적으로 나올 수 있는 평균의 분포에서, 지금 내가 갖고 있는 값인 95 보다 큰 값이 나올 수 있는 확률. 그것이 p-value 이다.**
        * 만약 그럴 확률이 매우 낮다면 우리는 귀무가설을 기각할 수 있게 된다. 왜냐 하면, 우리는, 우연히 발생할 가능성이 매우 희박한 사건이 실제로 발생했을 경우, 그것은 우연이 아니라고 생각하는 경향이 있고, p-value 역시 그와 같은 경향을 따른 것이기 때문이다
        * ex) 즉, 평균이 100, 분산이 30인 모분포에서 50개를 선택했을 때 평균이 95가 나오는 경우가 매우 드물다면, 아마도 내가 갖고 있는 데이터는 P에서 왔다고 말하기 조금 꺼려진다. 반대로 그럴 확률(A)이 0.65 라면, 그렇다면 이런 경우는 그리 어려운 일이 아니므로 그럴듯 하다.
        * 따라서 p-value가 너무 낮으면, 그렇게 낮은 확률의 사건이 실제로 일어났다고 생각하기 보다는 귀무가설이 틀렸다고 생각하게 된다. 그래서 귀무가설을 기각하고 대립가설을 채택하게 된다.
    
    * TIP : P-value에 관한 여러가지 오해들(본문 내용)
        ```
        1. p-value는 귀무가설이 참일 확률이 아니다 : 귀무가설이 참일 확률은 구할 수 없다.
        2. p-value는 통계값이 우연일 확률이 아니다 : p-value가 낮아도 귀무가설이 참일 수 있고, p-value가 높아도 귀무가설은 틀릴 수 있다.
        3. p-value는 귀무가설을 기각하면 안되는데 기각할 확률이다 : 길어서 뒤로 뺌.
        4. p-value는 반복실험을 했을 때 동일하지 않은 결론이 나오는 확률이 아니다 : 100 번을 sampling 하면 5번 정도는 p-value 0.05 에 걸리겠지. 이 경우, 95번은 귀무가설 채택, 5번은 기각, 이라는 것은 p-value 0.05 를 기준으로 그 이하는 좀 일어나기 어려우니까 그냥 귀무가설이 틀렸다고 하자, 라는 가정 때문이지 p-value 때문은 아니다.
        5. 1-(p-value)는 대립가설이 맞을 확률이 아니다 : p-value와 대립가설은 별로 관련이 없다. 순전히 '귀무가설이 맞다는 전제 하에' 나온 값이 p-value이고, p-value를 구함에 있어 대립가설은 그 어디에서도 작용하지 않는다.
        6. significance level은 p-value에 의해 결정되는 것이 아니다 : alpha는 연구자의 주관이며, 관례적으로 0.05, 0.01 을 사용할 뿐이지. 난 microarray 가 지저분하기 때문에 0.10 정도를 사용할 때도 있다.

        3. p-value는 귀무가설을 기각하면 안되는데 기각할 확률이다 : 아무래도 가장 혼란스러운 오해가 아닌가 싶다. 귀무가설을 잘못 기각했는지, 아니면 맞게 기각했는지는 확인할 수 없다.
        즉, 귀무가설을 잘못 기각했다는 것은 확률값이 아니다. 귀무가설이 맞다는 전제 하에 나온 분포에서 무엇인가를 하는 것이기 때문에, 그리고 p-value가 0.001 이에서 귀무가설을 기각했다고 해서 그것이 귀무가설이 맞음에도 불구하고 p-value가 낮았기 때문에 기각했다고는 말히기는 좀 어려운데 왜냐 하면 그 0.1%의 경우에 대해서 귀무가설이 사실은 맞은 가설인지를 확인할 방법이 없기 때문이다. 그리고 정의상 p-value는 그런 개념이 아니다.
        만약 애초에 귀무가설이 틀렸다고 해보자. 그런 상황에서도 여전히 p-value는 구해지는데, 그런 p-value가 과연 귀무가설을 잘못 기각한 확률이 되는가?
        아마도 3번과 같은 오류는, 귀무가설이 맞다는 전제 하에 모든 일이 이루어진다면 맞는 말이긴 한데, 애초부터 귀무가설이 틀릴 수도 있고, 그렇더라도 p-value는 여전히 구해지기 때문에 뭐라 말할 수 없게 되는 것이다.
        ```
    * 참고 : https://adnoctum.tistory.com/332
* `1종 오류` & `2종 오류`
    * 1종 오류(Type 1 Error) : 귀무가설(H0)이 참(True)일 때, 귀무가설(H0)을 기각(Reject)하는 경우 
    * 2종 오류(Type 2 Error) : 귀무가설(H0)이 거짓(False)일 때, 귀무가설(H0)을 기각하지 못하는 경우
    * 유의수준(alpha)을 높인다는 것은 2종 오류가 발생할 가능성을 낮춘다는 의미이다.
        * <-> 유의수준(alpha)을 낮춘다는 것은 2종 오류가 발생할 가능성을 높인다는 의미이다.
    * 귀무가설(H0)을 기각함으로써 상당한 비용이 발생하는 경우, 연구자는 1종 오류가 발생할 가능성을 최대한 줄이고자 노력할 것이다.
    * Example : 신약개발
        * 1종 오류 : 귀무가설(H0-효과x) 참(true)인데 기각함(효과가 없는데 효과가 있다고(H1 채택) 검정함) => 약이 효과가 없는데 불구하고 신약이 효과 있다고 결론 내림 -> 많은 비용(비싼 약값을 주고 효과없는 약을 구매할 확률)이 발생하므로 1종 오류를 줄여야 한다.
        * 2종 오류 : 귀무가설(H0-효과x) 거짓(false)인데 H0 채택함 -> 신약에 대한 효과가 존재하는데 효과 없다고 검정해버림.


<br><br>

* **딥러닝(Deep Learning)** 과 **머신러닝(Merchine Lerning)**, **인공지능(AI)** 의 구별법

* Categorical Variables & Vector 처리 방법
  * 1) Dummy variable(가변수)로 변경 후 사용 : One-hot Encoding
  * 2) Feature hashing (a.k.a the hashing trick)
  * 3) Encoding to ordinal variables
  * 4) Cat2Vec -> Categorical to Vector `Cat2Vec` 파이썬 라이브러리 사용
    * Embedding 같은 개념으로 처리
    * Word2Vec은 단어들의 유사할수록 가깝게 위치하게 되는데 Cat2Vec 역시 원래의 차원보다 더 낮은 차원으로 차원축소를 진행하면서 Sparse함을 없앨 수 있다. 

* 불균형 데이터(Imbalanced data) 처리 방법
  * `class weight` 옵션을 이용해 특정 class의 weight 업데이트 양 늘려주기
  * oversampling : ex) Smote를 이용해 Data preprocess 
  * undersampling : class가 2개 있는데 데이터가 충분히 많은 상태에서 불균형하다 하면 undersampling 하기.
  * Loss 변경 : ex) `Focal Loss`
  * Reinforcement Learning 사용 : ex) `Using Deep Q-Learning in the Classification of an Imbalanced Dataset`
  * keras-balanced-batch-generator 사용
    * 배치마다 class를 동일한 비율로 바꿔주는 generator
    * 코드 github : https://github.com/soroushj/keras-balanced-batch-generator

*※ 여러가지 용어*
===
* 적확도 : **SFA**
    * SFA = 1 – GAP(실제값 - 예측값)/FCST(예측값)

* 배깅과 부스팅
    * Boosting은 Bagging과 유사하게 초기 샘플 데이터를 조작하여 다수의 분류기를 생성하는 기법 중 하나지만 가장 큰 차이는 순차적(Sequential)방법이라는 것입니다. 앞서 살펴본 bagging의 경우 각각의 분류기들이 학습시에 상호 영향을 주지않고 학습이 끝난 다음 그 결과를 종합하는 기법이었다면, Boosting은 이전 분류기의 학습 결과를 토대로 다음 분류기의 학습 데이터의 샘플가중치를 조정해 학습을 진행하는 방법입니다. 

    * 장단점
        * 이러한 이유로 이전 학습의 결과가 다음학습에 영향을 주게 되고 부스팅 라운드를 진행할수록 m차원 공간의 분류경계선(Borderline)상의 데이터의 가중치가 증가하게 되는 결과를 가져오게 됩니다. 일반적으로 부스팅 알고리즘은 의사결정나무(Decision Tree)모형을 주로 사용하는 것으로 알려져 있고 과적합(Over fitting)에 강한 장점을 갖고 있습니다. 하지만 다른 앙상블 모형과 마찬가지로 분류결과에 대한 해석이 불가능하다는 단점을 갖고 있습니다.

* 엔트로피(Entropy)와 크로스 엔트로피(Cross-Entropy)의 쉬운 개념 설명
  * http://melonicedlatte.com/machinelearning/2019/12/20/204900.html
  * 위 링크 설명 매우 잘되어 있음.

  * 엔트로피는 불확실성의 척도입니다. 정보이론에서의 엔트로피는 불확실성을 나타내며, 엔트로피가 높다는 것은 정보가 많고, 확률이 낮다는 것을 의미합니다. 불확실성이라는 것은 어떤 데이터가 나올지 예측하기 어려운 경우라고 받아들이는 것이 더 직관적입니다.
  * 크로스 엔트로피 : q에 대하여 알지 못하는 상태에서, 모델링을 통하여 구한 분포인 p를 통하여 q 를 예측하는 것입니다. q와 p가 모두 들어가서 크로스 엔트로피라고 한다고 합니다.

* hash_map과 map의 차이점
  * 둘의 가장 큰 차이는 특정 키에 대한 값을 찾는 과정에서, hash_map은 이름 그대로 hash table을 이용해서 키-값 관계를 유지하며, map은 red-black tree 알고리즘을 이용합니다.

</br>


*※ 강화학습 관련*
===

* 마르코프 의사 결정
    * 마르코프 의사 결정 과정은 다음 상태 의 확률은 오직 현재의 상태 와 현재의 행동 에만 영향을 받고, 이전의 상태와 행동에는 영향을 받지 않는 마르코프 가정을 기반으로 합니다.
* 차감된 미래의 리워드(Discounted Future Reward)

</br>

*※ 사이트 및 논문*
===

* 도움되는 사이트
    * 딥러닝 간단한 설명 : https://yjjo.tistory.com/5
    * 간단한 딥러닝 연습문제 : https://leechamin.tistory.com/237?category=827905

* 논문
    * Stock market index prediction using artificial neural network
        * https://www.sciencedirect.com/science/article/pii/S2077188616300245
        * https://www.researchgate.net/publication/305746718_Stock_market_index_prediction_using_artificial_neural_network
    * Batch Normalization
        * https://arxiv.org/abs/1502.03167
        * https://blog.lunit.io/2018/04/12/group-normalization/
        * https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf
        * https://blog.airlab.re.kr/2019/08/Group-Normalization
        * https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/
        * http://sanghyukchun.github.io/88/

* 세미나 논문 읽을 LIST
    * http://www.materic.or.kr/community/board_anony/content.asp?idx=29&page=1&board_idx=&s_kinds=&s_word=&s_gubun=&listCnt=\
    * 알파고 논문 관련
        * https://roundhere.tistory.com/entry/%EC%95%8C%ED%8C%8C%EA%B3%A0
        * https://wegonnamakeit.tistory.com/17
        * https://blog.naver.com/sogangori/220668124217
        * https://www.youtube.com/watch?v=a4H-P10pVz4
    * Reinforcement Learning 관련
        * Playing Atari with Deep Reinforcement Learning
        * https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        * https://ddanggle.github.io/demystifyingDL
        * https://jamiekang.github.io/2017/05/07/playing-atari-with-deep-reinforcement-learning/
        * 기초 논문 리스트 
            * http://www.materic.or.kr/community/board_anony/content.asp?idx=29&page=1&board_idx=&s_kinds=&s_word=&s_gubun=&listCnt=
        * 이웅원님의 Fundamental of Reinforcement Learning
            * https://dnddnjs.gitbooks.io/rl/content/
            * 연관 사이트
                * https://www.intel.ai/demystifying-deep-reinforcement-learning/#gs.qrk9c6
                * https://sumniya.tistory.com/2?category=781573
        *  Sutton의 Introduction to Reinforcement Learning
           *  책 : http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
           *  강의 : https://www.youtube.com/watch?v=2pWv7GOvuf0
        * https://www.researchgate.net/publication/318798849_Deep_learning_for_stock_market_prediction_from_financial_news_articles
        * https://arxiv.org/pdf/1903.06478.pdf
        * https://arxiv.org/pdf/1909.12227.pdf
        * https://www.aclweb.org/anthology/U17-1001.pdf
<br>
* 강화학습 관련 논문 리딩 방향성
    * Q-MIX 관련 paper
    * reward function 관련 paper
    * stock maket trading 관련 paper
