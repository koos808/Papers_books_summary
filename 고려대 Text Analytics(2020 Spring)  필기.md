## *※ STEP 01-2 : Introduction to Text Analytics: Part2*
### 강의 영상 : https://www.youtube.com/watch?v=Y0zrFVZqnl4&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=3&ab_channel=KoreaUnivDSBA

---

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






