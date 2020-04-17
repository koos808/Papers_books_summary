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