## NLP 기초
### Part 1. NLP(Natural Language Processing, 자연어처리) 기초 다지기 위해 강의 수강

* https://www.youtube.com/watch?v=dKYFfUtij_U&list=PLVNY1HnUlO26qqZznHVWAqjS1fWw0zqnT
    * Bag of Words
    * n-gram
    * TF-IDF
    * 자연어처리의 유사도 측정 방법(거리측정, 코사인 유사도)
    * TF-IDF 문서 유사도 측정
    * 잠재의미분석(LSA - Latent Semantic Analysis)
    * Word2Vec
    * WMD 문서 유사도 구하기(word mover's distance)
    * RNN 기초(순환신경망 - Vanila RNN)
    * LSTM 쉽게 이해하기
    * Sequence to Sequence + Attention Model
    * Transformer(Attention is all you need)


#### 1. Bag of words
* 문장을 숫자로 표현하는 방법 중 하나.
* Sentence similarity : 문장의 유사도를 구할 수 있음.
* 머신러닝 모델의 입력 값으로 사용할 수 있다.(문자를 숫자형으로 바꿔주어서 입력값으로 사용 가능하다)
* 한계(Limitation)
    * `Sparsity` : If we use all English words for bag of words, the vector will be very long, but very few non zeros.
        * 실제 문장 하나를 표현할 때 0이 무수히 많기 때문에 계산량이 높으며 메모리도 많이 사용한다.
    * Frequent words has more power : 많이 출현한 단어는 힘이 세진다. 
    * Ignoring word orders : 단어의 순서를 철처히 무시한다. 단어의 출현 순서를 무시하기 때문에 문맥이 무시된다. 예시로 home run vs run home 같은 것으로 인식함.
    * Out of vocabulary : 보지 못한 단어들은 처리하지 못한다.

#### 2. n-그램(n-gram)
* n-그램은 연속적으로 n개의 토큰으로 구성된 것을 의미함. 토큰은 자연어 처리에서 보통 단어나 캐릭터로 얘기함.

* 1-gram == unigram
    * EX) fine thank you
    * Word level : [fine, thank, you]
    * Character level : [f,i,n,e, ,t,h,a,n,k, ,y,o,u]
* 2-gram == bigram
    * EX) fine thank you
    * Word level : [fine thank, thank you]  -> fine you는 붙어있지 않아서 토큰이 안됨.
    * Character level : [fi,in,ne,e , t,th,ha,an,nk,k , y,yo,ou]
* 3-gram == trigram
    * EX) fine thank you
    * Word level : [fine thank you]
    * Character level : [fin,ine,ne, et, th,tha,han,ank,nk, ky, yo,you]

##### `Why n-gram?`
* bag of words는 단어의 순서가 철저하게 무시되는 단점이 있는데 이를 n-gram으로 극복할 수 있다.
    * bag of words' drawback(단점)
    * EX) machine learning is fun and is not boring
    * `bag of words` : (machine,fun,...boring) -> (1,1,2,1,1,1,1)
    * `n-gram` : (machine learning, learning is, is fun, fun and, and is, is not, not boring ) -> (1,1,1,1,1,1,1)
    * not boring as token now! : not boring을 인식 잘하게 될 수 있다.

* Next word prediction : 다음 단어가 무엇이 올지 예측가능하다.
    * Naive Next word prediction
    * EX) how are you doing, how are you, how are they -> Trigram(how are you, are you doing, how are they) = (2,1,1)
    * how are ??? : ???을 예측하려면 어떤 단어를 예측할까? -> `you` [나이브한 방법일 때]
* Find Misspelling
    * Naive한 Spell checker 방법
    * EX) quality, quater, quit -> Bigram(qu,ua,al,li,it,ty,at,te,er,ui) = (3,2,1,1,2,1,1,1,1,1)
    * q`w`al => w가 잘못 되고 qual로 스펠링 체크해줄 수 있음(나이브한 방법)

#### 3. TF-IDF
* 용어
    * TF : Term(단어) Frequncy
    * IDF : Inverse Documnet Frequency
* TF-IDF를 왜 이용할까?
    * To find how `relevant` a term is in a document : 한 문서는 단어로 구성되어 있다. 각 단어별로 문서에 관한 연관성을 알고 싶을 때 TF-IDF를 이용한다.
    * 각 단어별로 문서에 대한 정보를 얼마나 갖고 있는지를 TF-IDF 수치로 나타낸 것이다.
* `Term Frequncy`
    * TF measures how frequently a term occurs in a document : 문서에서 단어가 몇번 출현했는지.
    * TF를 사용 시 가정
        *  If a term occurs more times than other terms in a document, the term has more relevance than other terms for the documnet. : 문서가 있을 때 한 단어가 여러번 출현했을 때 그 단어는 문서와 연관성이 높을 것이다.
    * EX1) "a new car, used car, car review"
    * word(a,new,car,used,review) -> TF(1/7,1/7,3/7,1/7,1/7) : car가 이 문장과 가장 연관성이 높은 것으로 보임.
    * TF measure의 치명적인 단점(drawback)
        * EX2) "a friend in need is a friend indeed"
        * word(a,friend,in,need,is,indeed) -> TF(2/8,2/8,1/8,1/8,1/8,1/8)
        * EX1과 동일하기 TF Score를 구했는데 중요하지 않은 a가 friend와 같이 가장 높았음. -> TF의 가정이 틀림. 따라서 IDF의 개념이 나오게 된다.
* `IDF(Inverse Documnet Frequency)`
    * `LOG(Total # of Docs / # of Docs with the term in it)` : 총 문장의 개수를 단어가 출현한 문장의 개수로 나누어 준것에 LOG를 취한것.
    * LOG(Total # of Docs / # of Docs with the term in it `+1`) : 때로는 0으로 나누는 것을 방지하기 위해 +1을 더해주기도 함(smoothing)
    * A : "a new car, used car, car review"\
    B : "a friend in need is a friend indeed"
    * ![tf-idf](./image/IF-IDF_Example.png)


#### 4. 자연어 유사도 측정(Euclidean Distance cosine similarity)

* 자연어 처리에서 유사도 측정하는 방법 2가지
    * 1.Euclidean Distance
    * 2.cosine similarity
    * 영상 참고

#### 5. TF-IDF 문서 유사도 측정

* How to get document similarity?
    * Cosine Similarity on Bag of Words
    * Cosine Similarity on TF-IDF with Bag of Words -> 이걸 쓰는게 좋음.
    * 영상 참고
* IF-IDF with Bag of words의 `장점`
    * Easy to get document similarity : 문서의 유사도를 구하기 쉽고 구현하기 쉽다.
    * Keep relevant words socre : 중요한 단어의 점수를 유지한다.
    * lower just frequent words score : 자주 출현하지만 여러 문서에 등장한다면 점수를 줄여준다.
* IF-IDF with Bag of words의 `단점`
    * Only based on Terms(words) : 단어만 본다. 단어의 유사성 같은건 안봄
    * Weak on capturing document topic : 단어는 알되 그 topic을 아는 것은 한계가 있음.
    * Weak handling synonym(different words but same meaning) : 이음동이의를 처리하기 힘듬.
* IF-IDF with Bag of words의 `단점`을 보안할 수 있는 방안
    * LSA(Latent Semantic Analysis)
    * Word Embeddings(Word2Vec, Glove)
    * ConceptNet - knowledge 그래프 사용

#### 6. 잠재 의미 분석(LSA)
* LSA similarity is based on `topic` : IF-IDF with Bag of words는 word(단어) 기반이지만 LSA는 topic 기반이다.
* 영상 참고

#### 7. WMD(Word mover's distance) 문서 유사도 구하기

* WMD는 Word2Vec의 유클리디안 거리를 사용한다.

* What is Word2Vec?
    * Word Embedding
    * Word similarity comes from the word's neighbor words : 한 문장에서 그 단어의 이웃들로 계산이 된다.
    * You also can easily download pre-trained word2vec
    * GoogleNews-vectors-negative300.bin.gz
* Word2Vec example
    * 영상 참고
* WMD 핵심 아이디어
    * Normalized Bag of Wors after stop words removal
* WMD 단점 
    * 계산 속도가 굉장히 느리다. -> RWMD(Relaxed WMD)에서는 조금 개선되서 $p^3logp\;에서\;p^2\;만큼\;빨라짐$; p는 문장 내 unique 단어 개수.

#### 8. Tensor란?
* tensor example in `NLP`
    * ex) (3,2,4)[3D tensor] = `(sample dimension,max length of sentence, word vector dimension)` : 3개의 샘플(문장)을 갖고 있는데, 문장 내 단어 개수는 2개이며, word들은 총 4개로 이루어져 있다.
* tensor example in `grayscale image`
    * (3,5,5)[3D tensor] = `(you have 3 images, 5 rows, 5 columns)` : 3개의 이미지를 갖고 있는데 이미지는 5x5 행열로 이루어져 있다
* tensor example in `rgb color image`
    * (3,5,5,3)[4D tensor] = `(you have 3 images, 5 rows, 5 columns, red-green-blue)` : 3개의 이미지를 갖고 있고 이미지는 5x5 행열로 이루어져 있는데 3개의 색으로 구성되어 있다.
* tensor example in `rgb color video`
    * (3,5,5,5,3)[5D tensor] = `(you have 3 frames, 5 images, 5 rows, 5 columns, red-green-blue)` : 총 3개의 비디오가 있고 비디오 안의 frame은 5개의 이미지로 구성되어 있다. 그리고 그 이미지는 5x5 행열로 이루어져있고 3개의 rgb 색이 있다.

#### 9. RNN(순환 신경망)
* 영상 참고

#### 10. LSTM
* 영상 참고

#### 11. Sequence to Sequence with Attention Model
* 영상 참고

#### 12.Transformer(Attention is all you need)
* 영상 참고

1. Transformer는 기존 encoder decoder architecture를 발전시킨 모델이다.
2. RNN을 사용하지 않는다.
3. RNN 기반 모델보다 학습이 빠르고 성능이 좋다.(Faster, Better!)

* How faster?
    * Reduced sequence computation
    * Parallelization(병렬)

`※ Encoder`

기존 인코더, 디코더의 주요 컨셉을 간직하되 RNN을 없애서 학습 시간을 단축시켰으며 성능도 올렸다.

* Positional encoding
    * 단어의 위치와 순서 정보를 활용하기 때문에 rnn을 사용했는데 rnn을 제거 했기 때문에 Positional encoding을 사용한다.
    * Positional encoding이란 인코더 및 디코더 입력값마다 상대적인 위치정보를 더해주는 기술이다.
    * 장점 1 : 항상 Positional encoding의 값은 -1~1 사이의 값이다.
    * 장점 2 : 학습 데이터 중 가장 긴 문장보다도 긴 문장이 운영 중 들어왔을 때 에러없이 상대적인 인코딩 값을 줄 수 있다.

* Self Attention
    * Quary, key, value => vector의 형태
    * Quary * key = Attention Score
        * Score가 높을수록 단어의 연관성이 높고 Score가 낮을수록 연관성이 낮다.
* Residual Connection followed by layer normalization
* Encoder Layer에 입력 vector와 출력 vector의 차원의 크기는 동일하다. -> 이는 즉, Encoder Layer를 여러개 붙혀서 사용할 수 있다는 뜻이다. Transformer는 Encoder Layer를 6개 붙혀서 만든 구조다.
* Transformer Encoder의 최종 출력 값은 6번째 인코더 레이어의 출력값이다.

`※ Decoder`
Decoder는 Encoder와 동일하게 6개의 Layer로 구성되어있다. 


* Label Smoothing -one hot encoding이 아님.

---
### Question
* word embedding이랑 character embedding을 같이 사용하는 이유 -> word embedding만 사용하면 새로운 단어가 들어왔을 때 얼타는 경우가 있는데, character emmbedding을 사용하면 모든 문자와 특수문자 모두를 사용하기 때문에 조금 더 정확도가 올라간다.
* uni-LSTIM(한방향)이 아니라 Bi-LSTM(양방향)으로 학습하는 이유? -> 앞으로도 읽어보고 뒤로도 읽어보고 다양한 방법으로 context를 이해하기 위함.

# 세미나 발표 준비

### 발표 논문 : XLNet: Generalized Autoregressive Pretraining for Language Understanding

* paper url : https://arxiv.org/pdf/1906.08237.pdf

* 참고 사이트
    * https://www.youtube.com/watch?v=koj9BKiu1rU&t=1327s
    * https://www.slideshare.net/SungnamPark2/pr175-xlnet-generalized-autoregressive-pretraining-for-language-understanding-152887456
    * https://ai-information.blogspot.com/2019/07/nl-041-xlnet-generalized-autoregressive.html
    * https://blog.pingpong.us/xlnet-review/
    * https://ratsgo.github.io/natural%20language%20processing/2019/09/11/xlnet/

* 목표 : 전부 다 읽고 XLNet 이해하기

### ※ XLNet 정리

참고 사이트 : https://ratsgo.github.io/natural%20language%20processing/2019/09/11/xlnet/

#### 기본 내용
---
* XLNet은 트랜스포머 네트워크(Vaswani et al., 2017)를 개선한 ‘트랜스포머-XL(Dai et al., 2019)’의 확장판 성격의 모델입니다. 
* 여기에서 XL이란 ‘eXtra-Long’으로, 기존 트랜스포머보다 좀 더 넓은 범위의 문맥(context)를 볼 수 있다는 점을 강조하는 의미로 쓰였습니다.

#### 퍼뮤테이션 언어모델 (Permutaion Language Model)
---
* Yang et al.(2019) 최근 임베딩 모델의 두 가지 흐름 : `오토리그레시브(AutoRegressive, AR) 모델`, `오토인코딩(AutoEncoding, AE) 모델`

* `오토리그레시브(AutoRegressive, AR) 모델`
  * `AR 모델`은 데이터를 순차적으로 처리하는 기법의 총칭을 뜻합니다. 
  * 이 관점에서 보면 `ELMo`나 `GPT`를 AR 범주로 분류할 수 있습니다. 두 모델 모두 이전 문맥을 바탕으로 다음 단어를 예측하는 과정에서 학습하기 때문입니다.
  * 아래 예제의 문장을 학습하는 경우 AE 모델은 단어를 순차적으로 읽어 나갑니다. <br><br>
  * Ex) <image src="image/AR model.jpg" style = "width:350px">

* `오토인코딩(AutoEncoding, AE) 모델`
  * `AE 모델`은 입력값을 복원하는 기법들을 뜻합니다. -> `y=f(x)=x`를 지향합니다.
  * 대표적인 AE 모델로는 BEAR가 있는데, BERT는 문장 일부에 노이즈(마스킹)을 주어서 문장을 원래대로 복원하는 과정에서 학습하는 모델입니다.
  * 즉, 마스킹 처리가 된 단어가 실제로 어떤 단어일지 맞추는데 주안점을 두는 것입니다.
  * 이런 맥락에서 BERT를 디노이징 오토인코더(Denoising Autoencoder)라고 표현하기도 합니다.
  * 디노이징 오토인코더란 노이즈가 포함된 입력을 받아 해당 노이즈를 제거한 원본 입력을 출력하는 모델입니다. <br><br>
  * Ex) <image src="image/AE model.jpg" style = "width:300px">

* ※ Yang et al.(2019)의 AE,AR 모델의 **문제점 제안**
  * AR 모델의 문제점
    * 문맥을 양방향(bidirectional)으로 볼 수 없는 태생적인 한계가 있다.
    * 이전 단어를 입력(Input)으로 하고 다음 단어를 출력으로 하는 언어모델을 학습할 때, 맞춰야 할 단어나 그 단어 이후의 `문맥 정보`를 미리 알려줄 수는 없기 때문이다.
    * 물론 ELMo의 경우는 모델의 최종 출력값을 만들 때 마지막 레이어에서 순방향(forward), 역방향(backward) LSTM 레이어의 계산 결과를 모두 사용하기는 합니다.
    * 그러나 `pre-train`을 할 때 순방향, 역방향 레이어를 `각각 독립적으로 학습하기 때문에` ELMo는 진정한 의미의 양방향(bidirectional) 모델이라고 말하긴 어렵습니다.

  * AE 모델의 문제점
    * BERT는 AE의 대표 양방향 모델입니다. 이는 마스크 단어를 예측할 때 앞뒤 문맥을 모두 살피며 성능 또한 좋았습니다. 하지만, AE 모델 역시 단점이 있습니다.
    * 가장 큰 문제는 <u>마스킹 처리한 토큰들을 서로 독립(independent)이라고 가정한다는 점</u>입니다. 이 경우 `마스킹 토큰들 사이에 의존 관계(dependency)를 따질 수 없게 됩니다.` <br><br>
    * <image src="image/bert3.jpg" style = "width:300px"> <br><br>
    * 위 사진은 Yang et al.(2019)이 논문에서 사용한 예시 문장('New York is a city')을 가지고 BERT의 학습 과정을 시각화한 것입니다. 영어 말뭉치에서 “New가 나온 다음에 York라는 단어가 나올 확률”과 “New가 나오지 않았을 경우에 York가 등장할 확률”은 분명히 다를 것입니다. 하지만 BERT 모델은 두 단어의 선후 관계나 등장 여부 등 정보를 전혀 따지지 않습니다. 그저 is a city라는 문맥만 보고 New, York `각각의 마스크 토큰을 독립적으로 예측합니다`.
    
    * 또한, BERT는 pre-train할 때 사용하는 마스크 토큰(mask token)은 파인 튜닝(fine-tuning) 과정에서는 사용하지 않습니다. fine-tuning과 다른 pre-train 환경을 구성하면 모델의 일반화(generalization) 성능이 떨어질 수 있다는 단점이 있습니다. 마지막으로 BERT는 긴 문맥을 학습하기 어렵다는 단점도 있습니다.

* ※ AR 모델 문제점 해결 : 퍼뮤테이션 언어모델 - 양방향(bidirectional) 문맥 고려 가능
  * <u>토큰을 랜덤으로 셔플한 뒤 그 뒤바뀐 순서가 마치 원래 그랬던 것인 양 언어모델을 학습하는 기법이다.</u> 
  * 아래 그림은 `발 없는 말이 천리 간다`를 퍼뮤테이션 언어모델로 학습하는 예시입니다. 모델은 `‘없는, 이, 말, 발, 간다’`를 입력받아 시퀀스의 마지막 단어인 ‘천리’를 예측합니다.

  * Ex) <image src="image/퍼뮤테이션 언어모델1.jpg" style="width:300px">
  * 이렇게 퍼뮤테이션을 수행하면 특정 토큰을 예측할 때 `문장 전체 문맥을 살필 수 있게 됩니다`. 즉, 해당 토큰을 제외한 문장의 부분집합 전부를 학습할 수 있다는 뜻입니다.
  
  * Ex) 발 없는 말이 천리 간다는 문장을 한번 더 퍼뮤테이션해서 이번엔 `발, 없는, 천리, 이, 말, 간다`가 나왔다고 가정하면, ‘천리’라는 단어를 예측할 때의 입력 시퀀스는 `발, 없는`이 됩니다.
  * 위 예제처럼 `"천리"`라는 단어를 맞추기 위해 모든 입력 시퀀스들을 조합할 수 있고, 거기엔 순방향 언어모델과 역방향 언어모델 모두 `퍼뮤테이션 언어모델의 부분집합`으로 들어가게 됩니다.
  * 다시 말해 퍼뮤테이션 언어모델은 시퀀스를 순차적으로 학습하는 AR 모델이지만 퍼뮤테이션을 수행한 토큰 시퀀스 집합을 학습하는 과정에서 `문장의 양방향 문맥을 모두 고려할 수 있게 된다는 이야기`입니다.

* ※ AE 모델 문제점 해결 : 퍼뮤테이션 언어모델 - 단어 간 의존관계를 포착
  * 퍼뮤테이션 언어모델은 AR이기 때문에 BERT 같은 AE 모델의 단점 또한 극복할 수 있다고 설명합니다.
  * 퍼뮤테이션 언어모델은 셔플된 토큰 시퀀스를 순차적으로 읽어가면서 다음 단어를 예측하며, 이전 문맥(New)을 이번 예측(York)에 활용합니다.
  * 따라서 퍼뮤테이션 언어모델은 예측 단어(Masking token) 사이에 독립을 가정하는 BERT와는 달리 단어 간 의존관계를 포착하기에 유리합니다.
  * 뿐만 아니라, pre-train 때 마스크하지 않기 때문에 pre-train & pine-tuning 간 불일치 문제도 해결할 수 있습니다.

  * Ex) <image src="image/퍼뮤테이션 언어모델3.jpg" style="width:300px">

* ※ 퍼뮤테이션 언어모델(permutaion language model) 학습 예시
  * <image src="image/퍼뮤테이션 언어모델 학습.jpg" style="width:250px">
  * ◎ Example 1 : [3,2,4,1]
  * 토큰 네 개짜리 문장을 랜덤으로 뒤섞은 결과가 그림 7처럼 [3,2,4,1]이고 셔플된 시퀀스의 첫번째 단어(3번 토큰)를 맞춰야 하는 상황이라고 가정해 봅시다.
  * 그러면 이 때 `3번 토큰 정보`를 넣어서는 안 됩니다. 3번 토큰을 맞춰야 하는데 모델에 3번 토큰 정보를 주면 문제가 너무 쉬워지기 때문입니다. 
  * 2번, 4번, 1번 토큰은 맞출 토큰(3번) 이후에 등장한 단어들이므로 이들 또한 입력에서 제외합니다. 결과적으로 이 상황에서 `입력값은 이전 세그먼트(segment)의 메모리(memory) 정보뿐입니다`. 메모리와 관련해서는 트랜스포머-XL(transformer-XL)에서 설명합니다.
  
  * ◎ Example 2 : [2,4,3,1]
  * <image src="image/퍼뮤테이션 언어모델 학습2.jpg" style="width:250px">
  * 같은 문장을 또한번 셔플했더니 [2,4,3,1]이고 이번 스텝 역시 3번 토큰을 예측해야 한다고 가정합시다. 그러면 3번 토큰 이전 문맥(메모리, 2번, 4번 단어)이 입력됩니다. 3번 토큰은 정답이므로 입력에서 제외합니다. 그림 8과 같습니다.
  
  * ◎ Example 3 & 4 : [1,4,2,3], [4,3,1,2]
  * <image src="image/퍼뮤테이션 언어모델 학습3.jpg" style="width:250px"> <image src="image/퍼뮤테이션 언어모델 학습4.jpg" style="width:250px">
  * 셔플 시퀀스가 [1,4,2,3]이고 3번 토큰을 맞춰야 한다면 입력벡터는 과거 문맥(메모리, 1번, 4번, 2번 단어)이 됩니다. 마찬가지로 [4,3,1,2]이고 3번을 예측한다면 입력값은 메모리, 4번 단어가 됩니다. 각각 그림 9, 그림 10과 같습니다.

* ※ 퍼뮤테이션 언어모델 학습 과정 1 : 원래 시퀀스의 어텐션 마스크 
  * 퍼뮤테이션 언어모델의 실제 구현은 토큰을 뒤섞는 게 아니라 `어텐션 마스크(attention mask)`로 실현됩니다. 
  * XLNet의 근간은 기존 트랜스포머 네트워크(Vaswani et al.,2017)이고, 그 핵심은 `쿼리(query), 키(key) 벡터 간 셀프 어텐션(self-attention) 기법`이기 때문입니다. 
  * 예컨대 토큰 네 개짜리 문장을 단어 등장 순서대로 예측해야 하는 상황을 가정해 봅시다. 이 경우 어텐션 마스크는 그림 11처럼 구축하면 됩니다.
  
  * <image src="image/원래 시퀀스 어텐션 마스크.jpg" style="width:300px">
  * 위 그림에서 좌측 행렬은 `셀프 어텐션을 수행할 때 소프트맥스 확률값에 적용하는 마스크 행렬`입니다. 여기서 마스크(Mask)란 소프트맥스 확률값을 0으로 무시하게끔 하는 역할을 한다는 뜻입니다. 소프트맥스 확률값이 0이 되면 해당 단어의 정보는 `셀프 어텐션에 포함되지 않습니다`. 회색 동그라미는 확률값을 0으로 만드는 마스크라는 뜻이며, 붉은색 동그라미는 확률값을 살리는 의미를 지닙니다.
  * 그림을 읽는 방법 : 마스크 행렬의 행은 쿼리 단어, 열은 키 단어에 각각 대응합니다. 그림 11처럼 토큰 순서대로 예측해야 하는 경우 1번 단어를 예측할 때는 자기 자신(1번 단어)을 포함해 어떤 정보도 사용할 수 없습니다. 2번 단어를 맞춰야할 때는 이전 문맥인 1번 단어 정보를 활용합니다. 마찬가지로 3번 단어는 1, 2번 단어, 4번 단어는 1, 2, 3번 단어 정보를 쓰게끔 만듭니다. GPT가 그림 11과 같은 방식으로 학습합니다.

* ※ 퍼뮤테이션 언어모델 학습 과정 2 : <u>셔플된 시퀀스의 어텐션 마스크</u>
  * 아래 그림 12는 퍼뮤테이션 언어모델이 사용하는 어텐션 마스크의 예시입니다.
 
  *  <image src="image/셔플된 시퀀스 어텐션 마스크.jpg" style="width:300px">
  
  * 셔플된 토큰 시퀀스가 [3,2,4,1]이라고 가정해 봅시다. 그러면 3번 단어를 맞춰야할 때는 어떤 정보도 사용할 수 없습니다. 2번 단어를 예측할 때는 이전 문맥인 3번 단어 정보를 씁니다. 마찬가지로 4번 단어를 맞출 때는 3번, 2번 단어를, 1번 단어를 예측할 때는 3번, 2번, 4번 단어 정보를 입력합니다.

* ※ 퍼뮤테이션 언어모델 단점 : 투-스트림 셀프 어텐션으로 해결 
  * 예컨대 단어가 네 개인 문장을 랜덤 셔플한 결과가 다음와 같고 이번 스텝에서 셔플 시퀀스의 세번째를 예측해야 한다고 해봅시다. ==> `[3, 2, "4", 1] , [3, 2, "1", 4]`
  * 이 경우 모델은 동일한 입력(3번, 2번 단어)을 받아 다른 출력을 내야 하는 모순에 직면합니다. Yang et al.(2019)는 이같은 문제를 해결하기 위해 `투-스트림 어텐션(two-stream self attention) 기법`을 제안했습니다. 

#### 투-스트림 셀프어텐션(Two-Stream Self Attention)
---
* ※ 정의
  * 투-스트림 셀프어텐션(Two-Stream Self Attention)은 `쿼리 스트림(query stream)과 컨텐트 스트림(content stream)` 두 가지를 혼합한 `셀프 어텐션 기법`입니다. 

* ※ 컨텐트 스트림(content stream)
  * 컨텐트 스트림은 기존 트랜스포머 네트워크와 거의 유사하다.
  * 