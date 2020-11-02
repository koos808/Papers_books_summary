## *※ STEP 01-2 : Introduction to Text Analytics: Part2*
### 강의 영상 : https://www.youtube.com/watch?v=Y0zrFVZqnl4&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=3&ab_channel=KoreaUnivDSBA

---
## TM Process 1 : Collection & Preprocssing

* Text Preprocessing Level 1 : **Sentence**
* 문서가 가장 상위의 개념이라고 봤을 때, 문서의 바로 하위 개념은 문단(Sentence)이다.
  * Correct sentence boundary is also important.
  * Sentence를 정확히 구분해내는 것은 굉장히 어려운 문제이다.
    * Sentence를 분리하기 위해 Rul-based Model을 사용할 수도 있고, konlp 등의 형태소 분석기를 사용할 수도 있으며 다양한 방법이 있다.

* Text Preprocessing Level 2 : **Token**
  * `Extracting meaningful`(worth being analyzed - 가장 적은 단위의 의미가 주어져 있는 단어를 Token이라 한다.) tokens(word, number, space, etc.)
  * 하이픈(-)이나 소유격("'s"), 공백 자르기 등의 token으로 나누는것에도 여러가지 어려운 문제가 있음.
  * `Power distribuion` in word frequencies
    * 빈번하게 사용되는 단어가 더 중요하다라는 의미는 text-analytics에서는 틀리다.
    * 빈번하게 사용되는 단어일수록 관사나 전치사 등의 문법적인 기능을 하되 의미, semantic 관점에서는 중요하지 않을 가능성이 높은 단어들이다. 그래서 이런 단어들은 `Stop-words`로 구분한다.
  * `Stop-words`
    * Words that <U>do not carry any information</U> -> 어떠한 분석 관점에서의 정보를 포함하지 않는 단어이다.
      * Mainly functional role
      * Stop-words 들은 없으면 논리적이지 않게 되지만 문장의 핵심적인 내용이 없어지는게 아니다. 핵심적인 내용은 살아있으니 자연어 처리 관점에서는 문제없다.
    * `Stemming`
      * 차원을 줄이는 관점.
      * 품사, 미래형, 과거형 등 서로 다른 형태의 단어들을 하나의 단어로 맞춰주는 작업을 의미함.
      * **Stemming is a process of transforming a word into its stem(normalized form)**
    * `Lemmatization`
      * 차원을 줄이는 관점.
      * Although stemming just finds any base form, which does not even need to be a word in the language, but lemmatization finds the actual root of a word.
      * => *Stemming*은 base form을 찾는 것이라면, *Lemmatization*은 해당하는 단어들의 품사를 보존하면서 **단어의 원형**을 찾는 것이 목적이다.
      * 
        ```
        [Word]         [Stemming]      [Lemmatization]
        Love            Lov             Love
        Loves           Lov             Love
        Loved           Lov             Love
        Loving          Lov             Love
        Innovation      Innovat         Innovation : 명사
        Innovations     Innovat         Innovation : 명사
        Innovate        Innovat         Innovate : 동사
        Innovates       Innovat         Innovate : 동사
        Innovative      Innovat         Innovative : 형용사
        ```
    * `Stemming`은 결과물 자체가 사전에 존재하지 않는 단어일 수도 있다는 단점이 있지만, 훨씬 더 원형을 기본으로 줄여나가기 때문에 결과물 자체 개수가 적다는 장점이 있다. 하지만 `Lemmatization`은 품사를 보존하기 때문에 결과물 개수가 많다.
    * 즉, 차원의 축소 관점에서 보면 `Stemming`이 좀 더 효율적인데 단어의 품사를 보존하는 관점에서는 `Lemmatization`이 더 효율적이다.

## TM Process 2 : Transformation

### Document representation
-> 문서를 어떻게 하면 연속형의 숫자 벡터로 표현할 것인가가 핵심!

* `Bag-of-words` : simplifying representation method for documents where a text is represented in a vector of an unordered collection of words
  * 하나의 문서는 그 문서에서 사용된 단어들의 빈도나 출연 여부로 표현하는 것.
* `Word Weighting`
  * 특정한 단어가 어떠한 문서에서 얼마만큼 중요한지에 대해서 가중치를 부여하는 것이다.
  * 가중치를 부여하는 것에서 가장 많이 사용했던 것이 `TF-IDF`이다.
    * tf(w) : term frequency (number of word occurrences in a document)
    * df(w) : number of documents containing the word (number of documents containing the word)

* One-hot-vector representation
  * The most simple & intuitive representation
  * Can make a vector representation, but similarities between words cannot be
preserved. 
  * => 가장 큰 단점은 두 단어 사이에 유사성이 보존될 수 없다는 점이다.

* `Word vectors` : distributed representation
  * 위의 유사성을 고려하지 못하는 단점을 보완하기 위해 표상(distributed representation)으로 나타냈다.
  * 단어를 n차원의 실수 공간 상에 맵핑을 해주는 방법론인데 이것이 `Word vector`이다.
  * Interesting feature of word embedding
    * Semantic relationship between words can be preserved
  * 즉, 충분한 corpus를 통해서 학습하게 되면 단어 사이의 유사성이 보존되는 분산 표상을 만들어 낼 수 있다.
  * 그래서 이러한 `Word vectors`를 직접 학습하는게 아니라 `Word2vec`, `fastText`, `GloVe`(Global Vectores For Word Representation) 등의 **Pre-trained Word Models**을 사용하던가, `Elmo`, `GPT`, `Bert` 같은 Model을 pre-trained 시킨 **Pre-trained Language Models**을 사용한다.
  * 위의 Pre-trained Models를 사용해서 다운스트림 테스크(Natural Language Inference, 분류 등)를 진행한다. 

## TM Process 3 : Dimensionality Reduction 

### Feature Selection/Extraction

* Feature subset <U>selection</U>
  * Select only **the best features** for further analysis : 특정한 목적에 걸맞는 가장 최적의 변수 집합을 선택하는 것이다.
  * Scoring methods for individual feature (for supervised learning tasks)
    * Information gain, Cross-entropy, Mutual information, Weight of evidence, Odds ratio, Frequency 등
  * 위와 같은 산출식을 이용해서 어떤 토큰이나 단어들이 우리가 원하는 태스크에 유의미한지를 판별하는 게 Feature subset selection이다. 

* Feature subset <U>extraction</U>
  * 기본적인 요건 : 주어진 데이터의 차원이 d이고, extraction된 데이터의 차원이 d'이면 반드시 원래 차원보다 작아야 한다.(`d>d'`)
  * Feature extraction : `construct a set of variables that preserve the information of the original data` by combining them in a linear/non-linear form
  * **Latent Semantic Analysis (LSA)**
  * 단어들을 3개의 matrix로 Decompose한 이후에 이중에서 2개의 매트릭스를 가져다가 차원을 축소하는 방법이다.
  * SVD in Text Mining : 문서나 단어를 축약하는 데 사용되기도 하는 방법이다.
  * Topic Modeling as a Feature Extractor
    * **Latent Dirichlet Allocation (LDA)**
    * Topic Modeling의 원래 목적은 unsupervised 관점에서 문서 집합 corpus를 관통하고 있는 주요 주제를 판별한다. 그 주요 주제는 두 가지의 결과물로 나타난다. 첫번 째는 문서 별로 주제의 비중이며, 두번째는 각각의 주제별로 토픽 별로 단어들이 얼마만큼 발생 빈도를 가지고 있을 것인가이다. 즉. 토픽별로 주요 핵심 키워드를 찾아내는 것이다.
    * (a) Per-document topic proportions는 document별로 Topic들이 얼마나 비중을 가지고 있는지를 나타내고, (b) Per-topic word distributions는 각각의 토픽별로 단어들이 얼마만큼 비중을 갖고 있는지 산출한 것이다. Feature extraction 관점에서는 (a)를 사용한다.
  * Document to vector(`Doc2Vec`)
    * Use a distributed representation for each document : 각각의 document 차원에서 distributed representation을 만들어 낸다.

## TM Process 4 : Learning & Evaluation

다운 스트림 태스크로서 우리가 원하는 최종적인 태스크를 학습하고 평가하는 단계이다.

#### ※ Similarity Between Documents
* Document similarity
  * Use the **cosine similarity** rather than Euclidean distance

#### ※ Learning Task 1: Classification
* Document categorization(Classification)
  * Sentiment Analysis

#### ※ Learning Task 2: Clustering
* Document Clustering & Visualization 

#### ※ Learning Task 3: Information Extraction/Retrieval

* Information extraction/retrieval
  * Find useful information from text databases
  * Examples: Question Answering
* Topic Modeling
* Latent Dirichlet Allocation(LDA)

## *※ STEP 02-1 : Text Preprocessing - Part 1*
### 강의 영상 : https://www.youtube.com/watch?v=NLaxlUKFVw4&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=4&ab_channel=KoreaUnivDSBA

---
## 1. Introduction to NLP
### Natural Language Processing
* Classical categorization of NLP
  * Phonology, Morphology, Syntax, Semantics, Pragmatics, Discourse
* An example of NLP
  * Lexical Analysis -> Syntax Analysis -> Semantic Analysis -> Pragmatic Analysis
* Rule-based 에서 machine-learning(deep-learning) approaches로 바뀜
  * Neural Machine Translation
  * End-to-End Multi-Task Learning
  * 

## 2. Lexical Analysis




## 3. Syntax Analysis




## 4. Other Topics in NLP


