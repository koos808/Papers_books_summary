※ Edwith - 논문으로 짚어보는 딥러닝의 맥
===========
</br>

## *※ STEP OT : 배울 사항 및 분야*
* 목차
    * 1. CNN - AlexNet, GoogleNet
    * 2. Regularization
    * 3. Optimization Methods
    * 4. RBM
    * 5. Denoising Auto Encoder
    * 6. Semantic Segmentation
    * 7. Weakly Supervised Localization
    * 8. Detection Methods
    * 9. RNN - LSTM with 한글
    * 10. Visual Q&A
    * 11. Super Resolution
    * 12. Deep Reinforcement Learning
    * 13. Sequence Generation
    * 14. Word Embedding - Word2Vec
    * 15. Image captioning
    * 16. Residual Network & Analyses
    * 17. Neural Style
    * 18. GAN :: Generative Adversarial Network
    * 19. GAN Series :: GT, GAN, GAN-CLS, GAN-INT, GAN-INT-CLS, DCGAN
    * 20. Logistic Regression
    * 21. MLP : Multi-Layer Perceptron
    * 22. Mixture Density Network
    * 23. Domain Adaptation :: Domain Adversarial Network
    * 24. VAE :: Variational Autoencoder
    * 25. Adversarial Variational Bayes
    * 26. One-Shot learning
    * 27. Metirc Learning
    * 28. Memory Network
    * 29. Uncertainty in Neural Networks

## *※ STEP 1 : Convolutional Neural Network(CNN)의 기초*
* 핵심 키워드
    * 
    ```
    Convolutional Neural Network (CNN)
    Convolutions
    Subsampling
    Convolutin layer
    Fully connected layer
    ```
* CNN :: Convolutional Neural Network
* 절차
    * Input -> (Convolutions) -> Convolution feature maps -> (Subsampling) -> (Convolutions) -> (Subsampling) -> ... -> Full Connection(= Dense Layer = Fully Connected Layer)
    * Subsampling : 이미지 안에서 더 작은 영역을 얻는 작업(이미지 축소 및 빈공간 줄이기)
    * CNN = Convolution + Subsampling + Full Connection
* CNN 구조
  * CNN = ( Convolution + Subsampling[ex) Pooling] ) + Full Connection
* Convolution과 Subsampling은 feature extraction의 역할을 한다. 이미지를 classifies(분류)하기 위해서 추출된 feature를 가지고 Connected layer를 사용한다.
  * feature extraction은 이미지에서 중요한 feature들을 뽑아내어 이미지를 구분하는데 사용한다.
  * 최종 Output은 class에 속할 확률들이 나온다.
  
* CNN이 Powerful한 이유
  * Local Invariance :: 국소적으로 비슷하고 차이가 없다.
    * **Loosely speaking** - Convolution filter가 전체 이미지를 모두 돌아다닌다.
        * Local Invariance : Loosely speaking, as the convolution filters are `sliding` over the input image, the exact location of the object we want to find does not matter much.
    * **Compositionality**  - 구성
      * There is a hierarchy(계층 구조) in CNNs. It is GOOD!
* Convolution 연산
  * Image(5x5), Convolved Feature(3x3)
  * 즉, 내가 가지고 있는 Convolutional Filter 모양과 Convolution을 하는 픽셀들이 얼마나 비슷한지를 나타내게 하는 것이 Convolution 연산이다.
  * 비슷하면 Convolutional Feature map의 값이 크게 나온다. 또한 가장 좋은 성능을 내는 Convolutional Filter 모양은 학습을 통해 찾는다.
* Zero-padding
  * 가장자리에서도 Convolution할 수 있도록 이미지 가장자리에 0을 추가한다.
  * `n_out = (n_input + 2*n_padding - n_filter) + 1`
    * 
    ```
    ex) 1.n_input pixel : 1x5
        2.n_filter : 3
        3.n_padding : 1
        4.n_output pixel : (5 + 2*1 - 3) + 1 = 5
    ```
* Stride
  * 몇 칸마다 convolution을 계산할 것인지 지정할 수 있다.
  * If stride size equals the filter size, there will be `no overlapping`.

* Channel
  * Out channel : 내가 지금 갖고 있는 Convolutional Filter의 개수
  * In channel : input image channel 수
  * `parameter의 수`는 적을 수록 좋음
   ```
   ex) input : 4(height)x4(width)x3(channel)
      filter : 3x3x3
      out channels : 7
      what is number of parameters in this convolution layer?
      => 189 = 3 x 3 x 3 x 7

   ```
* 전체적인 간단한 과정
  * `Input` - pixel(3x3x1) -> `Conv1` - 3x3 convolution(64 filters) -> Convolutional Feature map(28x28x64) -> add bias(28x28x64) -> Relu(Recitified Linear unit)(28x28x64) -> Max pooling(14x14x64) -> Reshape(re-ordering)(14x14x64) -> Fully Connected Layer(10개의 숫자) -> one-hot coding(10개의 vector중에서 가장 큰 숫자의 index 라벨을 사용함)
  * Convolutional Layer의 parameter 수 -> 3x3(filter)x64(filter 개수) + 64(bias 개수) = 640
  * Fully Connected Layer의 parameter 수 -> 14x14x64x10(output)+10(bias 개수) = 125,440
  * Convolutional Layer의 parameter 수보다 Fully Connected Layer를 정의할 때 필요한 parameter 수가 훨씬 많다. 파라미터의 수가 많아지면 안좋으므로 앞단에 Convolution Layer를 많이 붙히고 Fully Connected Layer를 간소화 시키던가, 없애던가해서 네트워크 전체를 정의하는 파라미터의 수를 줄이려고하는게 요즘 뉴럴렛의 트렌드임.
* Epoch / Batch size Iteration
  * One epoch : one forward and backward pass of `all training data` 
  * Batch size : the number of training examples in `one forward and backward pass` 
  * One iteration : number of passes
  * If we have 55,000 training data, and the batch size is 1,000. Then, we need 55 iterations to complete 1 epoch.

## *※ STEP 2 : 4가지 CNN 살펴보기: AlexNET, VGG, GoogLeNet, ResNet*
* 핵심 키워드
    * 
    ```
    CNN(Convolutional Neural Network)
    Conv2D
    AlexNet
    VGG
    GoogLeNet
    ResNet
    ```
* Layer를 Deep하게 쌓으면서 동시에 성능을 잘 나오게 하기 위한 테크닉들이다.
* AlexNet(ILSVRC 2012 1등)
    * What is the number of parameters? 
        * parameter의 수 : 11x11x3x48+48(channel)
    * Why are layers divided into two parts?
        * Gpu의 낮은 성능 때문!
    * `Relu 사용`
    * LRN(Local Response Normalization) 사용
        * regulization 테크닉 중 하나이다. Out put으로 나온 convolutional feature map 중에서 일정 부분만 높게 activation한다. 즉, convolution feature map 중에서 일정 부분만 높은 값을 갖고 나머지는 낮은 값을 갖도록 만드는 역할을 한다.
        * It implements a form of `lateral inhibition(측면 억제)` insppired by real neurons.
    * Regularization in AlexNet
        * Main objective is to reduce `overfitting`.
        * More details will be handled in next week.
        * In the `AlexNet`, two regularization methods are used.
            * `Data augmentation` : data를 늘리는 것. Flip augmentation(물체 좌우반전) & Crop(이미지 부분으로 나누기) + Flip augmentation. 하지만, 숫자와 같은 이미지들은 Flip augmentation을 하면 안되므로 하려는 이미지를 파악하고 Data augmentation을 적용해야한다.
                * Original Image(256x256) -> Smaller Patch(224x224) : This increases the size of the training set by a `factor of 2048(32x32x2(좌우반전))`. Two comes from horizontal reflections. 즉, 데이터를 뻥튀기 시켜서 학습을 시켰기 때문에 성능이 좋아졌다고 얘기했음.
                * Original Patch(224x224) -> Altered Patch(224x224) : 또한, `Color variation`을 적용했다. `Color variation`이란 것은 다음과 같다. RGB Image이기 때문에 그냥 Noise 성능이 아니라, 각각의 RGB Channel에 어떤 특정값을 더하게 된다. 더하는 특정값이란 것은 학습데이터에서 해당 RGB가 얼마나 많이 변했는지를 학습을 통해 알아내고 이 값에 비례해서 RGB 값을 넣게 된다. 예를 들어서 `Data augmentation`을 할 때, 빨간색에 Noise를 많이 넣게 되면 전혀 다른 라벨의 데이터가 나오기 때문에 학습데이터에서 허용할 수 있는 만큼의 noise만 넣어야 한다.
                * Probabilistically, not a single patch will be same at the training phase! (a `factor of infinity!`)
            * `Dropout` : 일반적인 Dropout은 해당 Layer에서 일정 퍼센트만큼의 노드를 0으로 만들어 준다. 하지만 여기서는 단지 output에 0.5만큼을 더했다. <- 이렇게 쓴거는 AlexNet 논문이 처음이자 마지막이다.
                * Original dropout sets the output of each hidden neuron with certain probability.
                * In this paper, they simply multiply the outputs by 0.5.
* VGG 
    * 매우 간단하다. Convolution은 모두 stride를 1로, 3x3을 Convolution에 활용했다.
    * Convolution은 3x3을 활용했으며, stride는 1로 하며, max pooling과 average pooling을 통해서 space한 정보를 줄이게 된다.
    * 3,1,1(feature map size 유지)로 모든 Layer를 만들고, Block이란 개념을 도입함.
    * Block을 사용해서 vanishing gradient 문제를 피할 수 가 있었음. end-to-end가 아니라서 부분 부분 좋은것들만 찾아서 합쳤기 때문. 
* GoogLeNet(ILSVRC 2014 1등)
    * 22 Layers Deep Network
    * Efficiently utilized computing resources, `Inception Module` : `Inception Module`을 알면 GoogLeNet을 다 이해한 것이다!
    * Significantly outperforms previous methods on ILSVRC 2014
    * `Inception Module`이란? 
        * AlexNet처럼 동일한 모양의 네트워크가 갈라진게 아니라, 1x1 convolutions, 3x3 convolutions, 5x5 convolutions 등의 다른 역할을 하는 convolutions들을 filter concatenation을 하여 방향순으로 쌓임.
        * filter concatenation이란 것은 Filter를 채널 방향으로 더한 것임.
        * `one by one convolution`을 추가함으로써 channel의 수를 중간에 한번 줄이고 이 네트워크를 정의하는 파라미터의 수를 줄일 수 있다. 즉, Layer가 한번 더 추가했는데도 파라미터의 수가 줄어든다. 정보를 축약도 하면서 파라미터 수를 줄인다.
        * 5x5로 보는 이유는 처음 학습할 때 크게 크게 보기 위해서임. base가 되는 것을 찾기 위해.
    * GoogLeNet은 `Inception Module`이 반복된 구조로 이루어져 있음. 1x1 convolution을 통해서 채널을 줄임으로써 전체 파라미터 수를 줄였음.
    * Conclusion
        * Very Clever idea of using `one by one convolution` for `dimmension reduction`!
        * `Inception Module`의 또 다른 차별점은 여러개로의 갈림길이 있기 때문에 더 다양한 정보들에서 추출할 수 있음.
        * 즉, GoogLeNet은 `Inception Module`과 `one by one convolution`을 가지고 Network를 만들었으며, 이를 통해 더욱 Deep한 Network를 만들 수 있으면서도 성능을 올릴 수 있었다.
        * GoogLeNet이 VGG보다 DEEP하면서도 파라미터 수가 절반 이상 적다.
        * 서로 다른 `receptive field`를 만들기 위해서 Image를 바로 convolutions을 하는게 아니라 1x1 convolutions, 3x3 convolutions, 5x5 convolutions로 각각 해보고 그것들을 concatenation한다. 그렇게 concatenation convolution feature map 위에 다시 1x1 convolutions, 3x3 convolutions, average pooling 등을 해주며 다시 `concatenation`한다. 이런식을 계속 반복하면서 output단의 input image 밑의 `receptive field`를 굉장히 다양하게 만들어 준다. 또한 `one by one convolution`을 통해서 `channel dimmension reduction`을 해주면서 Layer를 정의하기에 필요한 파라미터의 수를 줄인 것이 GoogLeNet 논문의 핵심이다.
        * inception module과 보조 classifier가 GoogLeNet이 한 일임.
        * 보조 classifier를 사용해서 vanishing gradinent가 발생해도 아래까지 weight가 업데이트 될 수 있도록 만들었음. 즉, signal이 아래까지 갈 수 있도록 만들었음.
        * 1x1 conv, 3x3 conv, 5x5 conv 다해보고나서 `채널적으로 중복된 정보가 많기 때문에 중복 정보를 줄이기 위해 1x1 conv(filter)를 사용한다(액기스만 가져온다).` -> conv 하고 싶은거 다하고, 나중에 좋은거 선택하는 방식으로 했음.
* Inception v4
    * 최근에 파라미터를 줄이기 위해서 어디까지 노력했냐의 산물
    * `Inception v4` model에서는 `Inception Module`에서 나오는 5x5 같은 convolutions이 더이상 나오지 않는다. receptive field를 늘리는 입장에서는 3x3 convolutions을 두번하던가 5x5 convolutions을 한번 하는 것과 동일하다.
    * 왜 잘되는 지는 알 수 없음.
* ResNet
    * 역시나 파라미터를 줄이기 위해 `BottleNeck architecture`를 사용했으며 `residual connection`이란 것을 사용했다.
    * ResNet 논문에서는 152 Layers Network로 구성되어 있다. 또한 동일한 Network가 여러 가지 대회에서 1등을 했다는 것은 이 방법론이 굉장히 범용적이며 다양한 곳에 활용될 수 있다는 의의를 주었다. 즉, 기존 코드에 `residual connection`이란 것을 추가하면 성능이 향상되는 의의가 있다.
    * 152 Layer 이상 되면 그게 그거일 것이다~~. 흔들리는 것만 잡자.
    * 논문에서의 문제 제기
        * Is deeper Network always better?(Deep한 Network가 항상 좋나?)
        * What about vanishing/exploding gradients?
            * -> Better initialization methods/batch normalization/ReLU 때문에 vanishing/exploding gradients 문제가 상대적으로 덜 중요해 졌다.
        * Any other problems?
            * Overfitting?
            * No -> `Degradation problem` : more depth but lower performance.
            * Overfitting과 Degradation의 차이점 : trainning error가 낮아지고 accuracy가 높아질 때, test accuracy가 계속 떨어질 때 Overfitting이라고 한다. 즉 Overfitting의 정의는 trainning data를 너무 잘 맞추려고 한 나머지 test data를 못맞추게 되는 것을 의미한다. Degradation은 무엇이냐면 trainning과 test data에서 잘되는데 성능이 잘 안나오는 것을 의미한다. 
    * Degradation problem을 해결하기 위해 Residual learning building block이란 것을 만들었음. 아이디어는 매우 간단함. 입력이 들어오고 출력이 있을 때 입력을 출력에 더한다. 이 때의 유일한 제약조건은 입력과 출력의 dimension이 같아야 하는 것이다. 어떤 target과 현재 입력사이의 차이만 학습을 하겠다는 것이 `Residual learning building block`이라 한다.
    * Why residual? (Residual이 왜 좋냐)   
        * We `hypothesize` that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. - residual mapping을 사용하면 더 좋을 것이라고 가정을 해봤는데 적용도 쉽고 실제로 잘 되더라~. 수학적인 background에서 나온 것이 아니라..
        * `Shortcut connections` are used.
        * "The extremely deep residual nets are `easy` to optimize." -> **easy**
        * "The deep residual nets can **easily** enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks."
    * ResNet의 가장 큰 단점은 same dimension.
    * Deeper bottle architecture
        * `Inception Module`에서 `one by one convolution`을 사용해서 파라미터를 줄이고 줄어든 파라미터를 가지고 convolution을 한뒤 concatenation 해주었다. ResNet도 비슷하게  `one by one convolution`으로 채널을 줄이고 convolution을 한뒤 `one by one convolution`을 다시 해준다(`차이점`). 즉, ResNet은 <u>Input(256 dimension) -> Dimension reduction(1x1,64) -> Convolution(3x3,64) -> Dimension increasemnet(1x1, 256)</u> 순으로 해주는데, 왜 마지막에 `one by one convolution`을 다시 해줬냐면 입력을 출력에 더해주기위해(same dimension 때문에) 다시 256채널로 복원해야 했기 때문이다. 즉 Dimension increasement가 Inception module과의 차이점이며 `Deeper bottle architecture`라고 한다.
    * ResNet 논문의 의의
        * 40개 정도의 Layer에서 생겼던 Degradation 문제를 100단 정도의 Layer에서 Degradation 문제가 발생하도록 밀었다고 볼 수 있음. 하지만 여전히 Layer 개수가 1000개가 넘어가면 Degradation 문제가 발생한다.
    * 객체 검출에서도 많이 사용함.
* ResNext
* PolyNet
## *※ STEP 3 : Overfitting을 막는 regularization*
* 핵심 키워드
    * 
    ```
    Regularization
    Overfitting
    ```
* Regularization Main purpose is to avoid `OverFitting`.
    * Overfitting이란 것은 학습데이터를 너무 믿는 나머지 테스트 데이터를 잘 맞추지 못하는 것을 의미함.
    * OverFitting
        * Literally, Fitting the data more than is warranted.
        * Things get worse with `noise!`
    * Noise 
        * Stochastic Noise ::: Comes form `random measurement error`(관측 에러)
        * Deterministic Noise ::: `Cannot model` this type of error -> 이런 에러는 모델링 할 수 없다.
        * Noise에는 위 두개의 Noise가 섞여있기 때문에 완벽히 decomposition할 수 없다.
        * mathematically
            * `VC dimension` - complexity of the algorithm(ex. 뉴럴렛의 Layer 수)
            * in-sample error - Training error
            * out-of-sample error - Test error
            * in-sample error와 out-of-sample error의 차이를 보편적으로  `Generalization performance`라고 함. `Generalization performance`가 높아지는게 진짜 제일 중요하다!!
            * `Generalization error`가 커지면 Over fitting이 나오게 되는 것임. 즉, VC dimension이 커지면(뉴럴렛을 복잡하게 할 수록) overfitting이 나올 확률이 커지게 된다.
    * Preventing OverFitting?
        * **Approach 1** : Get `more` data - 가장 중요!! 데이터가 부족하다면 data agumentation을 활용해 데이터를 뻥튀기 시켜야 한다.
        * **Approach 2** : Use a model with the right `capacity` - 적절한 능력을 갖는 모델을 활용해야 한다.
        * **Approach 3** : `Average` many different models (Ensemble) - 앙상블을 활용하는 것! 이게 바로 Bagging임. 앙상블을 사용하면 일반적으로 성능이 1~2%정도 향상된다.
        * **Approach 4** : Use DropOut, DropConnect, or BatchNorm - 위 3개를 해보고나서 테크닉(technique)들이 들어가게 된다.
    * Limiting the Capacity
        * Capacity를 줄인다는 것은 결국 네트워크 사이즈를 줄이는 것도 있지만 `Early stopping`도 Capacity에 속한다.
        * `Architecture` : Liit the number of hidden layers and units per layer
        * `Early stopping` : Stop the learning before it overfits using validation sets
        * `Weight-decay` : Penalize large weights using penalties or constraints on their squared values (L2 penalty) or absolute values (L1 penalty) - 학습하는 파라미터를 너무 크게 설정하고 싶지 않은 것! Weight가 너무 커지는 것을 방지하는 것을 추가함.
    * DropOut
        * DropOut increases the generalization performance of the neural network by `restricting` the model capacity! - 한 Layer가 있을 때 그 Layer의 노드를 랜덤으로 몇개 꺼버리는 것(학습 할 때만 0으로 만들기, 테스트할 때는 모두 사용).
        * 장점은 매번 mini batch마다 다른 아키텍쳐를 학습하는 효과가 있음.
    * DropConnect
        * Instead of turning the neurons off (DropOut), DropConnect `disconnects` the connections between neurons. - Layer의 값을 0으로 만드는 것이 아니라 weight를 끊어버리는 것임(0으로 주는 것).
    * Batch Normalization(중요!) -> 왠만한 문제에서 가능한 다 사용하면 됨.
        * mini-batch learning을 할 때, batch마다 normalization을 해서 statics들을 맞추는 작업을 해주는 것.
        * Benefits of BN
            * 1. Increase learning rate(learning rate를 늘릴 수 있다.) - Interval covariate shift를 줄일 수 있다. dataset들 간의 다른 statistics 때문에 일반적으로 learning rate가 너무 크면 학습이 되지 않는다. 하지만, Batch Normalization은 이러한 문제를 정규화시키기 때문에 learning rate를 높여서 학습할 수 있다.
            * 2. Remove Dropout - Dropout을 안써도 된다.
            * 3. Reduce L2 weight decay - L2 weight decay를 안써도 된다.
            * 4. Accelerate leaning rate decay - learning rate를 빨리해도 학습이 잘된다.
            * 5. Remove Local Response Normalization - Local Response Normalization를 안써도 된다.
    * Conclusion
        * 어떤 문제를 풀던 간에 Overfitting은 발생하기 때문에 Regularization은 항상 고려해야 한다. 아래는 순서.
        * 1.`data argumentation` -> data 늘리기
        * 2.`Layer Size`를 늘려가면서 학습해보기 & 데이터가 적을때는 적당한 `Network architecture`를 잡는게 진짜 중요하다.
* Book review - Ian Goodfellow's books - `http://www.deeplearningbook.org/` ::: Regularization chapter 7
    * List
    ```
    1. Parameter Norm Penalties
    2. Dataset Augmentation
    3. Noise Robustness : to input, weights, and output
    4. Semi-Supervised Learning = learning a "representation"
    5. Multitask learning
    6. Early Stopping
    7. Parameter Tying and Parameter Sharing
    8. Sparse representation
    9. Bagging and Other Ensemble Methods
    10. Dropout
    11. Adversarial Training
    ```
    * Parameter Norm Penalties
        * Many regularization approaches are based on limiting the `capacity` of models by adding a parameter norm penalty to the objective function.




