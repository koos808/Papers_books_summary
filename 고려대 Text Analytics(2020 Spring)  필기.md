# *※ STEP 01-2 : Introduction to Text Analytics: Part2*
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

---
# *※ STEP 02-1 ~ 02-2: Text Preprocessing - Part 1 & Part 2*
### 강의 영상 1): https://www.youtube.com/watch?v=NLaxlUKFVw4&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=4&ab_channel=KoreaUnivDSBA
### 강의 영상 2): https://www.youtube.com/watch?v=5gt1KvkkOlc&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=5&ab_channel=KoreaUnivDSBA

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
<br><br>

## 2. Lexical Analysis (어휘 분석)
- 단어나 토큰 수준의 의미를 보존할 수 있는 가장 최소한의 수준에서 분석하는 것을 의미한다.
- 일정한 순서가 있는 character들의 조합을 토큰으로 변환하는 것을 의미한다.
  
* Goals of lexical analysis
  * Convert a sequence of characters into a sequence of tokens, i.e., meaningful character strings.

* Process of lexical analysis
  * ✓ Tokenizing : 문서 토크나이징
  * ✓ Part-of-Speech (POS) tagging : 각 토큰이 문장에서 어떠한 형태소(명사, 동사 등) 가졌는지 확인
  * ✓ Additional analysis: named entity recognition(NER : 객체명 인식(사람, 날짜 등)), noun phrase recognition(명사구 인식), Co-reference(she, him 등 지시 대명사 같은 것을 의미), Basic dependencie, sentence split, chunking, etc.

#### Lexical Analysis 1 : Sentence Splitting (문장 구분)
* Sentence is very important in NLP, but it is `not critical` for some Text Mining tasks(Topic Model 같은 것).

#### Lexical Analysis 2 : Tokenization (문장 구분)
* Text is split into basic units called Tokens
  * ✓ word tokens, number tokens, space tokens, …
* Even tokenization can be difficult

#### Lexical Analysis 3 : Morphological Analysis (형태소 분석)
* Morphological Variants: `Stemming` and `Lemmatization`
  * `Stemming`
    * Stemming : Commonly used in Information Retrieval
    * Stemming 단점 : 언어에 종속적일 수 있으며(Ruleds are language-dependent) 스테밍 결과가 해당하는 language에 존재하지 않을 수 있다(computers -> comput). 그리고 서로 다른 단어가 하나의 stem으로 구분될 수 도 있다.(army, arm -> arm)
  * `Lemmatization`
    * Lemmatization is the process of deriving the base form, or lemma, of a word from one of its inflected forms. This requires a morphological analysis, which in turn typically requires a lexicon(사전,어휘).
    * Lemmatization 장점 : 품사를 보존하며 실제로 존재하는 단어이다.
    * Lemmatization 단점 : Stemming 보다 복잡하며 느리다.
    * 의미 분석이 중요할 때는 Stemming 보다 Lemmatization이 좀 더 적합하다.

#### Lexical Analysis 4 : Part-of-Speech(POS) Tagging (형태소 분석)
* Part of speech (POS) tagging
  * ✓ Given a **sentence X**, predict its <U>part of speech sequence Y</U>
    * Input : Token
    * Output : 가장 적합할 것 같은 POS Tagging
* Different POS tags for the same token
  * ✓ I love you. → “love” is a verb
  * ✓ All you need is love. → “love” is noun
* Pasing은 문장구조를 찾아내는 것이고, POS 태깅은 형태소를 찾아내는 것(토큰에 대해 품사를 할당하는 게 주목적)이다.

* POS Tagging Algorithms
  * Fundamentals(기초) : POS-Tagging generally requires
    * Training phase : 문장별로 태깅이 된 corpus가 필요하다.
    * Tagging algorithm
  * ✓ **Pointwise prediction**: predict each word individually with a classifier (e.g. `Maximum Entropy Model`, SVM)
  * ✓ Probabilistic models
    * **Generative sequence models**: Find the most probable tag sequence given the sentence (Hidden Markov Model; HMM) => 순차적으로 태깅을 할당한다.
    * **Discriminative sequence models**: Predict whole sequence with a classifier (Conditional Random Field; CRF) => 한꺼번에 일괄적으로 태깅을 할당하는 모델이다.
  * ✓ Neural network-based models
    * 최근에는 BERT 같은 Pre-trained model을 사용
    * Many to Many Task
    * Hybrid model: LSTM(RNN) + ConvNet + CRF

#### Lexical Analysis 5 : Named Entity Recognition(NER : 객체명 인식(사람, 날짜 등))

* a subtask of information extraction that seeks to `locate and classify elements in text into pre-defined categories` such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.

* Approaches for NER: Dictionary/Rule-based
  * 장점 : 빠르고 간단하다.
  * 단점 : 리스트 자체를 관리하는게 어렵다.
* Approaches for NER: Model-based
  * MITIE, CRF++, Convolutional neural networks etc.

* **BERT for Multi NLP Tasks**
  * Google Transformer


---
# *※ STEP 02-3 : Text Preprocessing - Part 3*
### 강의 영상 : https://www.youtube.com/watch?v=DdFKFqZyv5s&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=6&ab_channel=KoreaUnivDSBA


## 3. Syntax Analysis (구문 분석)
* Syntax Analysis
  * ✓ Process of analyzing a string of symbols conforming to the rules of a formal grammar

* `Parser`
  * ✓ An algorithm that computes a structure for an input string given a grammar => Input string으로 특정한 grammar에 걸맞게 변환하는 알고리즘이 `Parser`이다.
  * ✓ All parsers have *two fundamental properties*
    * ▪ `Directionality` : the sequence in which the structures are constructed (e.g., top-down or bottom-up)
    * ▪ `Search strategy` : the order in which the search space of possible analysis explored (e.g., depth-first, breadth-first) -> 탐색하는 전략
 
* Not a single parsing tree due to language `ambiguity(모호성)`
* Lexical ambiguity(어휘 모호성)
  * ✓ One word can be used for multiple parts of speech
    * => 같은 단어가 다른 의미로 사용될 수 있음.
  * ✓ Lexical ambiguity causes structural ambiguity

* Structural Ambiguity(구조적 모호성)

<br>

## 4. Other Topics in NLP

### ※ Language Modeling

* Probabilistic Language Model(확률적 언어 모델)
  * ✓ Assign a probability to a sentence (not POS tags, but the sentence itself)
    * => 특정한 sentence에 대해서 확률을 매기는 것이다.
    * 이를 통해 문장이 그럴 듯한 문장인가를 판단하고자 하는 것이다.
* Applications
  * ✓ Machine Translation
    * P(`high` wind tonight) > P(`large` wind tonight)
    * 위 문장에서 large라는 단어보다 high라고 쓰는게 더 그럴듯한 단어이므로 확률 또한 높다.
  * ✓ Spell correction
    * The office is about fifteen minuets from my house
    * P(about fifteen `minutes` from) > P(about fifteen `minuets` from)
    * 오타인 경우 확률이 낮음
  * ✓ Speech recognition
    * P(I saw a van) >> P(eyes awe of an)
    * STT 측면과 영어 관점에서 오른쪽 문장은 말이 되지 않기 때문에 확률이 낮다.
  * ✓ Summarization, question-answering, etc.

*  Probabilistic Language Modeling
  * ✓ Compute the probability of a sentence or sequence of words
  * 결합 확률 분포로 표현 : P(W) = P(w_1, w_2, .., w_n)

* Markov Assumption
  * ✓ Consider only k previous words when estimating the conditional probability
    * W_1이 등장할 확률 P(W_1)이나 W_1이 나왔을 때 W_2가 나올 확률 P(W_2|W_1)을 구하는 것은 어렵진 않으나, 10개의 문장에서의 P(W_10|W_1,...W_9)의 확률을 구하는 것은 어려울 것이다.
  * ✓ Simplest case: Unigram model
    * 위의 어려운 점을 가장 간단하게 해결할 수 있는게 `Unigram model`을 만드는 것이다.
    * 각각의 단어가 독립적으로 발생했다를 가정하자는 모델이다.
    * ![Unigram Model](./image/Unigram_Model.png)
    * Markov Assumption을 통해서 단순화를 시키는 작업을 한다.
    * 문장들이 문장 같지가 않다.

* Bigram model
  * ✓ Condition on the previous word
  * 컨디션은 바로 이전 단어에만 영향을 받는다.
  * ![Bigram Model](./image/bigram_Model.png)
  * 역시나 문장들이 문장 같지가 않다.

*  N-gram models
  * ✓ Can extend to <U>trigrams, 4-grams, 5-grams</U>
     * In sufficient model of language because language has long-distance dependencies
     * “`The computer` when I had just put into the machine room on the fifth floor `crashed.`”
   * ✓ We can often get away with N-gram model

---
# *※ STEP 4 : Text Representation 1 - Count-based Representations *
### 강의 영상 : https://www.youtube.com/watch?v=DMNUVGbLp-0&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=7&ab_channel=KoreaUnivDSBA

* INDEX
  * Bag of Words
  * Word Weighting
  * N-Grams

## 1. Bag of Words
* Bag-of-words: Term-Document Matrix
  * Binary representation : 몇번 등장했냐가 아니라 등장했냐 안했냐를 binary로 Check 
  * Frequency representation : 몇번 등장한지 빈도를 Check

* Bag of words Representation in a Vector Space
  * Vector representation does not consider the ordering of words in a document
    * 순서 고려 x
    * John is quicker than Mary = Mary is quicker than John in BOW representation
  * We cannot reconstruct the original text based on the term-document matrix
    * term-document matrix를 통해서 original text를 reconstruct를 할 수 없다.(반대로는 가능함.)
* Stop Words
  * What are stop words?
    * Words that `do not carry any information`.
    * Stop Words를 제거하여 불필요한 단어를 제거한다.
      * SMART stop words list, MySQL Stop words list 같은 것들이 있음.

## 2. Word Weighting
특정한 단어가 특정한 문서에 대해서 얼마만큼 중요한지 판단하는 것은 가중치 부여 측면에서 살펴보면 된다.

* Word Weighting: Term-Frequency (TF)
  * Term frequency $tf_{t,d}$
    * t는 term, d는 document이다.
    * term이 개별 document에서 얼만큼 나왔는지를 count. 해당 term이 특정 document에서 얼마만큼 중요한지를 나타낼 수 있다고 함.

* Word Weighting: Document Frequency (DF)
  * Document Frequency $df_t$
    * The number of documents in which the term t appears.
    * t라는 term이 전체 corpus 중에서 몇 개의 문서에서 등장했는지. 
    * 빈번하게 등장하지 않는 단어가 일반적인 단어들보다는 특정 문서의 중요도가 높을 가능성이 높다라는 얘기다.
    * We should give a high weight for `rare terms` <U>than common terms</U>

* Word Weighting: Inverse Document Frequency (IDF)
  * Inverse document frequency $idf_t$
  * $idf_t = log_{10}(N/df_t)$ : N은 Corpus에서의 문서의 개수, df_t는 해당하는 텀의 document frequency이다.

* Word Weighting: `TF-IDF`
  * TF는 커야하고, IDF는 낮아야 한다.
  * TF-IDF weight of a term is the product of its tf weight and its idf weight
  * <image src="image/TF-IDF.png" style="width:500px">
  * 해당 Term이 특정 document에 얼마나 중요하냐는 질문에 대한 정량적인 대답이다.
  * Best known weighting scheme in information retrieval
  * Increases with `the number of occurrences within a document`
  * Increases with `the rarity of the term in the collection`
  * 단점 : **Very high dimensional & Sparseness**

* The most commonly used TF-IDF in general
  * <image src="image/TF-IDF2.png" style="width:500px">

## 3. N-Grams

* N-Gram-based Language Models in NLP
  * <image src="image/n-gram.png" style="width:500px">


---
# *※ STEP 5 : Text Representation II Distributed Representations *
### 강의 영상 : https://www.youtube.com/watch?v=bvSHJG-Fz3Y&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=8&ab_channel=KoreaUnivDSBA

Lecture 5: Text Representation II Distributed Representations

* INDEX
  * Word-level : NNLM
  * Word-level : Word2Vec
  * Word-level : GloVe
  * Word-level : Fasttext
  * Sentence/Paragraph/Document-level
  * More Things to Embed?

### ✓ Distributed Representation: Word Embedding

* Word Embedding
  * The purpose of word embedding is `to map the words in a language into a vector space` so that <U>semantically similar words are located close to each other</U>.
  * 단어를 특정한 vector space로 매핑시키는 것인데, 의미적으로 유사한 단어는 서로 가까운 공간상에 위치하도록 한다.
  * Word vectors: one-hot vector
    * The most simple & intuitive representation : 가장 단순하면서 직관적인 representation은 `one-hot vector`이다.
    
    






    




