# 1. R-CNN(Regions with CNN features) :
##### 논문 : Rich feature hierarchies for accurate object detection and semantic segmentation
##### 참고 사이트 1 : https://yeomko.tistory.com/13?category=851298
##### 참고 사이트 2 : https://jaehyeongan.github.io/2019/10/10/R-CNN/

- 영향력: 인용 횟수가 무려 11000회에 달하며, 이후에 이어지는 R-CNN 시리즈들의 시작을 연 논문입니다.

- 주요 기여: CNN을 사용하여 object detection task의 정확도와 속도를 획기적으로 향상시켰습니다.

- 성능: Pascal VOC  2010을 기준으로 53.7%이며, 이미지 한 장에 CPU로는 47초, GPU로는 13초가 걸립니다.

* 중요 Keyword : Region Proposal, Seletive Search, 2000개의 bounding box, 227x227 size(resize=warp), 4096 차원의 특징 벡터,
Non-Maximum Supperssion, IoU(Intersection over Union) => 기준 0.5, Bounding Box Regression

* 찾아볼 용어 : fine tune, ground truth, mAP(object Detection의 정확도 측정 지표)

* 알아 두면 좋은 정보
    * 1.우선 `Obejct Detection`이란 이미지가 무엇인지 판단하는 `Classification`과 이미지 내의 물체의 위치 정보를 찾는 `Localization`을 수행하는 것을 말한다.
    * 2.저자는 해당 모델을 R-CNN(Regions with CNN features)이라고 명시하였으며, 그 이유는 CNN과 Region proposal이 결합되었기 때문이라고 한다.
    * 3.Image Classification과 다르게 detection은 이미지내에서 객체를 localizing하는 것이 요구되는데 이를 위해, 논문의 모델은 `sliding-window` 방식을 적용하였고, 높은 공간 해상도(high spartial resolution)을 유지하기 위해 5개의 Convolutional 레이어를 적용하였다.


<br/>
<br/>

* R-CNN의 전반적인 흐름
    * ![](./image/R-CNN의_전반적인_흐름.png)
    * ![](./image/R-CNN의_전반적인_흐름2.png)



* 논문 Abstract - 핵심 인사이트
    * 1.객체를 localize 및 segment하기 위해 bottom-up방식의 `region proposal(지역 제안)`에 Convolutional Neural Network를 적용
    * 2.domain-specific fine-tuning을 통한 supervised pre-training을 적용

* R-CNN 프로세스
    * 1.Input 이미지로부터 2,000개의 독립적인 `region proposal`을 생성
    * 2.CNN을 통해 각 proposal 마다 고정된 길이의 `feature vector를 추출`(CNN 적용 시 서로 다른 region shape에 영향을 받지 않기 위해 fixed-size로 이미지를 변경)
    * 3.이후, 각 region 마다 `category-specific linear SVM`을 적용하여 classification을 수행

* R-CNN Object Detection 알고리즘 수행 절차
    * 1.Input 이미지에 Selective Search 알고리즘을 적용해 물체가 있으란한 박스 2,000개를 추출한다.
    * 2.추출한 모든 박스를 227x227 크기로 resize(warp)한다. 이 때 박스의 비율 등은 고려하지 않는다.
    * 3.미리 image-Net 데이터를 통해 학습시켜놓은 CNN을 통과시켜 4096 차원의 특징 벡터를 추출한다.
    * 4.추출된 벡터를 가지고 각각의 클래스(Object의 종류)마다 학습시켜놓은 SVM Classifier를 통과시킨다.
    * 5.Bounding Box Regression을 적용해 박스의 위치를 조정한다.

### object detection의 3가지 모듈
1. category-independent한 `region proposals`를 생성
2. 각 region으로부터 feature vector를 추출하기 위한 `large CNN`
3. classification을 위한 `linear SVMs`

### 중요 Key Point
* 1.`Region Proposal`
    * ![](./image/Selective_Search.png)

    * 주어진 이미지에서 물체가 있을법한 위치를 찾는 것
    * 카테고리 독립적인 region proposal을 생성하기 위한 방법은 여러가지가 있는데 해당 논문에서는 이전 detection 작업들과 비교하기 위하여 Selective Search라는 최적의 region proposal를 제안하는 기법을 사용하여 독립적인 region proposal을 추출하였다.
    * `Selective Search`
        * R-CNN은 룰 베이스 알고리즘을 적용해 `2천개`의 물체가 있을법한 박스를 찾습니다.
        * Selective Search는 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을 법한 박스를 추론하게 됩니다.
        * R-CNN의 Region Proposal 과정 역시 뉴럴 네트워크가 수행하도록 발전했습니다.
        * 프로세스 
            ```
            1.이미지의 초기 세그먼트를 정하여, 수많은 region 영역을 생성
            2.greedy 알고리즘을 이용하여 각 region을 기준으로 주변의 유사한 영역을 결합
            3.결합되어 커진 region을 최종 region proposal로 제안

            ```

* 2.`Feature Extraction`
    * 추가 : 우선 위에서 언급한 Selective Search를 통해 도출 된 각 region proposal로부터 CNN을 사용하여 `4096차원`의 feature vector를 추출한다. 이후, feature들은 5개의 convolutional layer와 2개의 fully connected layer로 전파되는데, 이때 CNN의 입력으로 사용되기 위해 각 region은 `227x227 RGB의 고정된 사이즈`로 변환되게 된다.
    * Selective Search를 통해서 찾아낸 2천개의 박스 영역은 `227x227 크기로 리사이즈` 됩니다.(warp)
    * 그리고 Image Classification으로 `미리 학습되어 있는 CNN 모델`을 통과하여 `4096 size의 특징 벡터를 추출`합니다.
    * 미리 학습되어 있는 모델이란 저자들이 `이미지넷 데이터(ILSVRC2012 classification)`로 미리 학습된 CNN 모델을 가져온 다음, `fine tune`한 것입니다.
    * fine tune을 할 때 실제 Object Detection을 적용할 데이터 셋에서 `ground truth`에 해당하는 이미지들을 가져와 학습시켰습니다.
    * 그리고 Classification의 마지막 레이어를 Object Detection의 클래스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로 맞췄습니다.
* 정확도 측정 지표
    * `mAP` 사용
    * `정리` : 미리 이미지 넷으로 학습된 CNN(AlexNet)을 가져와서, Object Detection 용 데이터 셋으로 `fine tuning` 한 뒤, `selective search 결과로 뽑힌 이미지들로부터 특징 벡터를 추출`합니다.
* 추가
    * Classification에 최적화된 CNN 모델을 새로운 Detection 작업 그리고 VOC 데이터셋에 적용하기 위해 오직 VOC의 region proposals를 통해 SGD(stochastic gradient descent)방식으로 CNN 파라미터를 업데이트 한다. 이후 CNN을 통해 나온 feature map은 SVM을 통해 classification 및 bounding regreesion이 진행되게 되는데, 여기서 SVM 학습을 위해 NMS(non-maximum suppresion)과 IoU(inter-section-over-union)이라는 개념이 활용된다.
* 3.Classification
    * CNN을 통해 추출한 벡터로 각각의 클래스 별로 `SVM Classifier`를 학습시킵니다.
    * 주어진 벡터를 놓고 이것이 해당 물체가 맞는지 아닌지를 구별하는 Classifier 모델을 학습시킵니다. 그런데 왜 CNN Classifier를 놔두고 SVM을 사용할까? -> 저자의 답변 : "그냥 CNN Classifier를 쓰는 것이 SVM을 썼을 때 보다 mAP 성능이 4%가 낮아졌다. 이는 아마도 fine tuning 과정에서 물체의 위치 정보가 유실되고 무작위로 추출된 샘플을 학습하여 발생한 것으로 보입니다."
    * 이후 R-CNN 에서는 SVM을 붙혀서 학습시키는 기법이 더 이상 사용되지 않는다.
* 4.`NMS(Non-Maximum Suppression)`
    * ![](./image/Non-Maximum-Suppression.png)
    * Classification 단계에서 SVM을 통과하여 `각각 박스들은 어떤 물체일 확률 값(Score)`을 가지게 되었습니다. -> 여기서 의문점 : `2천개 박스가 모두 필요한 것인가?` -> 물론 `NO NO`
    * 동일한 물체에 여러 개 박스가 쳐져 있다면, 가장 스코어가 높은 박스만 남기고 나머지는 제거하고, 이 과정을 Non-Maximum Suppression이라 합니다.
    * 이 때 서로 다른 두 박스가 동일한 물체에 쳐져 있다고 `어떻게 판별`할 수 있을까요? -> 여기서 `IoU(Intersection over Union) 개념`이 적용됩니다. 쉽게 말하면 두 박스의 `Area of Overlap(교집합)/Area of Union(합집합) 값`을 의미하며 두 박스가 일치할 수록 1에 가까운 값에 나오게 됩니다.
    * * ![](./image/IoU.png)
    * 논문에서는 Iou가 `0.5보다 크면` 동일한 물체를 대상으로 한 박스로 판단하고 Non-Maximum Suppression을 적용합니다.
* 5.`Bounding Box Regression`
    * 위의 단계를 거치며 물체가 있을 법한 위치를 찾고, 해당 물체의 종류를 판별할 수 있는 Classification Model을 학습시켰습니다. 하지만 위의 단계까지만 거치면 Selective Search를 통해서 찾은 박스 위치는 상당히 부정확하게 됩니다.
    * 따라서 성능을 향상시키기 위해 `박스 위치를 조정해주는 단계`를 거치는데 이를 `Bounding Box Regression`이라 합니다. 
    * 각 SGD iteration마다 32개의 positive window와 96개의 backgroud window 총 128개의 배치로 학습이 진행된다.
* 5-1. 하나의 박스에 대한 수식
    * $P^i = (P_x^i,P_y^i,P_w^i,P_h^i)$ 여기서 x,y는 이미지의 중심점이고 x,h는 각각 너비(width)와 높이(height)입니다.
    * Ground Truth에 해당하는 박스는 $G = (G_x,G_y,G_w,G_h)$ 로 표기할 수 있습니다.
    * 여기서 목표는 P에 해당하는 박스를 최대한 G에 가깝게 이동시키는 함수를 학습하는 것입니다.
    * 박스가 인풋으로 들어왔을 떄, x,y,w,h를 각각 이동 시켜주는 함수들은 $d_x(P),\;d_y(P),\;d_w(P),\quad and\;d_h(P)$ 로 표현할 수 있습니다.
    * x,y는 점이기 때문에 이미지의 크기에 상관없이 위치만 이동시켜주면 되지만, 너비와 높이는 이미지의 크기에 비례해서 조정을 시켜줘야 합니다. 위의 특성을 반영해 P를 이동시키는 함수의 식을 짜보면 다음과 같습니다.
    * $\hat G_x= P_wd_x(P) + P_x$\
    $\hat G_y= P_hd_y(P) + P_y$\
    $\hat G_w= P_wexp(d_w(P))$\
    $\hat G_h= P_hexp(d_h(P))$
    * 학습을 통해서 얻고자 하는 함수는 위의 d 함수입니다. 저자들은 이 d 함수를 구하기 위해서 앞서 CNN을 통과할 때 pool5 레이어에서 얻어낸 특징 벡터를 사용합니다. 그리고 함수에 학습 가능한 웨이트 벡터를 주어 계산합니다.
    * d 함수 식 : $d_*(P) = W_*^T\;\phi_5(P)$\
    Loss function(MSE + L2 normalization) -> $w_\star = \underset{\hat w_\star}{argmin}\sum_{i}^N (t^i_*\;-\;W_*^T\;\phi_5(P^i))^2\,+\,\rVert\hat w_\star \rVert^2$ 로 나타낼 수 있다.
    * t는 P를 G로 이동시키기 위해서 필요한 이동량을 의미한다.
        * $t_x = (G_x-P_x)/P_w$\
        $t_y = (G_y-P_xy/P_h$\
        $t_w = log(G_w/P_w)$\
        $t_h = log(G_h/P_h)$
    * 정리를 해보면 CNN을 통과하여 추출된 벡터와 x, y, w, h를 조정하는 함수의 웨이트를 곱해서 바운딩 박스를 조정해주는 선형 회귀를 학습시키는 것입니다.

* R-CNN의 문제점
    * R-CNN의 가장 큰 문제는 복잡한 프로세스로 인한 과도한 연산량에 있다. 최근에는 고성능 GPU가 많이 보급 되었기 때문에 deep한 neural net이라도 GPU연산을 통해 빠른 처리가 가능하다. 하지만 R-CNN은 selective search 알고리즘를 통한 `region proposal 작업` 그리고 `NMS 알고리즘 작업` 등은 `CPU 연산`에 의해 이루어 지기 때문에 굉장히 많은 연산량 및 시간이 소모된다.
    * 속도 저하의 가장 큰 병목 구간은 selective search를 통해서 찾은 2천개의 영역에 모두 CNN inference를 진행하기 때문입니다.
    * 또한 SVM 예측 시 region에 대한 classification 및 bounding box에 대한 regression 작업이 함께 작동하다 보니 모델 예측 부분에서도 연산 및 시간이 많이 소모되어 real-time 분석이 어렵다는 단점이 있다.
    * 이와 같은 문제점을 해결하기 위해서, 추후 프로세스 및 연산 측면에서 보완된 모델이 나오게 되는데 그것이 바로 Fast R-CNN과 Faster R-CNN이다.

* R-CNN에서 학습이 일어나는 부분
    * 1.이미지 넷으로 이미 학습된 모델을 가져와 fine tuning 하는 부분
    * 2.SVM Classifier를 학습시키는 부분
    * 3.Bounding Box Regression


# 2. SPP-Net(Spatial Pyramid Pooling Network):
##### 논문 : Rich feature hierarchies for accurate object detection and semantic segmentation
##### 참고 사이트 1 : https://yeomko.tistory.com/14?category=851298

- 영향력: ResNet으로 유명한 Kaming He가 1 저자로 인용 횟수만 3600회에 달합니다.
- 주요 기여:  입력 이미지 크기와 상관없이 CNN을 적용할 수 있도록 하는 Spatial Pyramid Pooling 기법을 제안하였습니다.

### 중요 Key Point
기존의 CNN 아키텍쳐들은 모두 입력 이미지가 고정되어야 했습니다. (ex. 224 x 224) 그렇기 때문에 신경망을 통과시키기 위해서는 이미지를 고정된 크기로 크롭하거나 비율을 조정(warp)해야 했습니다. `하지만 이렇게 되면 물체의 일부분이 잘리거나, 본래의 생김새와 달라지는 문제점이 있습니다.` <u>여기서 저자들의 아이디어가 시작합니다.</u>

```
입력 이미지의 크기나 비율에 관계 없이 CNN을 학습 시킬 수는 없을까? 
```
![](./image/SPPNet_핵심_아이디어.png)

Convolution 필터들은 사실 입력 이미지가 고정될 필요가 없습니다. sliding window 방식으로 작동하기 때문에, 입력 이미지의 크기나 비율에 관계 없이 작동합니다. `입력 이미지 크기의 고정이 필요한 이유는 바로 컨볼루션 레이어들 다음에 이어지는 fully connected layer가 고정된 크기의 입력을 받기 때문입니다.` 여기서 Spatial Pyramid Pooling(이하 SPP)이 제안됩니다.

```
입력 이미지의 크기에 관계 없이 Conv layer들을 통과시키고,
FC layer 통과 전에 피쳐 맵들을 동일한 크기로 조절해주는 pooling을 적용하자!
```

* <U>저자들이 주장한 이 아이디어의 장점</u>
    1. 입력 이미지의 크기를 조절하지 않은 채로 컨볼루션을 진행하면 `원본 이미지의 특징을 고스란히 간직한 피쳐 맵을 얻을 수 있다.`
    2. 또한 `사물의 크기 변화에 더 견고한 모델을 얻을 수 있다`는 것이 저자들의 주장입니다.
    3. 이는 Image Classification이나 Object Detection과 같은 여러 테스크들에 `일반적으로 적용할 수 있다`는 장점이 있습니다.

* 전체적인 알고리즘
    1. 먼저 전체 이미지를 미리 학습된 CNN을 통과시켜 피쳐맵을 추출합니다.
    2. Selective Search를 통해서 찾은 각각의 RoI들은 제 각기 크기와 비율이 다릅니다. 이에 SPP를 적용하여 고정된 크기의 feature vector를 추출합니다.
    3. 그 다음 fully connected layer들을 통과 시킵니다.
    4. 앞서 추출한 벡터로 각 이미지 클래스 별로 binary SVM Classifier를 학습시킵니다.
    5. 마찬가지로 앞서 추출한 벡터로 bounding box regressor를 학습시킵니다. 

* 본 논문의 가장 핵심
    * Spatial Pyramid Pooling을 통해서 각기 크기가 다른 CNN 피쳐맵 인풋으로부터 고정된 크기의 feature vector를 뽑아내는 것에 있다. 그 이후의 접근 방식은 R-CNN과 거의 동일다.

* `SPP(Spatial Pyramid Pooling)` - 공간 피라미드 풀링
    ![](./image/Spatial_Pyramid_Pooling.png)

    * Conv Layer를 거쳐 추출된 feature map을 Input으로 받고 이를 미리 정해져 있는 영역으로 나누어 줍니다. 예를 들어 4x4(16), 2x2(4), 1x1(1) 세가지 영역이 있는데 각각을 하나의 피라미드라고 칭합니다. 즉, 3개의 피라미드를 설정한 것이며 피라미드 한 칸을 bin이라고 합니다. 예를 들어 입력이 64 x 64 x 256 크기의 피쳐 맵이 들어온다고 했을 때, 4x4의 피라미드의 bin의 크기는 16x16이 됩니다.
    * 이제 각 bin에서 가장 큰 값만 추출하는 max pooling을 수행하고, 그 결과를 쭉 이어붙여 줍니다. 입력 피쳐맵의 체널 크기를 k, bin의 개수를 M이라고 했을 때 `SPP의 최종 아웃풋은 kM 차원의 벡터`입니다. 위의 예시에서 k = 256, M = (16 + 4 + 1) = 21 이 됩니다.
    * `정리해보면 입력 이미지의 크기와는 상관없이 미리 설정한 bin의 개수와 CNN 채널 값으로 SPP의 출력이 결정되므로, 항상 동일한 크기의 결과를 리턴한다고 볼 수 있습니다.` 실제 실험에서 저자들은 1x1, 2x2, 3x3, 6x6 총 4개의 피라미드로 SPP를 적용합니다.

* `Object Detection에의 적용`
    * R-CNN은 Selective Search로 찾은 2천개의 물체 영역을 모두 고정 크기로 조절한 다음, 미리 학습된 CNN 모델을 통과시켜 feature를 추출합니다. 때문에 속도가 느릴 수 밖에 없습니다. `반면 SPPNet은 입력 이미지를 그대로 CNN에 통과시켜 피쳐 맵을 추출한 다음, 그 feature map에서 2천개의 물체 영역을 찾아 SPP를 적용하여 고정된 크기의 feature를 얻어냅니다.` 그리고 이를 FC와 SVM Classifier에 통과시킵니다.
    ```
    ※ R-CNN : Selective Search[Region Proposal] -> Crop/Warp 2K ROI -> CNN inferenc 2K RoI[Feature Extract] -> SVM[Classify]
    ※ SPPNet : Selective Search[Region Proposal] -> CNN inference 1 org image[Feature Extract] -> SPP 2K RoI -> SVM[Classify]
    ```
    ![compare](./image/SPPNet_vs_R-CNN.png)

* 개선한 부분
    * SPPNet은 기존 R-CNN이 모든 RoI에 대해서 CNN inference를 한다는 문제점을 획기적으로 개선했다.

* 한계점
    1. end-to-end 방식이 아니라 학습에 여러 단계가 필요하다. (fine-tuning, SVM training, Bounding Box Regression)
    2. 여전히 최종 클래시피케이션은 binary SVM, Region Proposal은 Selective Search를 이용한다.
    3. fine tuning 시에 SPP를 거치기 이전의 Conv 레이어들을 학습 시키지 못한다. 단지 그 뒤에 Fully Connnected Layer만 학습시킨다.


# 3. Fast R-CNN(ICCV 2015)
##### 논문 : Fast R-CNN - Ross Girshick , Microsoft Research


##### 참고 사이트 1 : https://yeomko.tistory.com/15?category=851298

- Fast R-CNN 논문을 보면 SPPNet 논문을 많이 참고한 것을 확인할 수 있다.

- 영향력: R-CNN 저자인 Ross가 1 저자로 인용 횟수만 8000회에 달한다.
- 주요 기여: `CNN fine tuning, boundnig box regression, classification을 모두 하나의 네트워크에서 학습시키는 end-to-end 기법을 제시했다.` 추후 이어지는 Faster R-CNN에 많은 영향을 주었다.
- 결과: SPPNet보다 3배 더 빠른 학습 속도, 10배 더 빠른 속도를 보이며 Pascal VOC 2007 데이터 셋을 대상으로 mAP 66%를 기록한다.

SPPNet에서는 기존 RCNN이 selective search로 찾아낸 모든 RoI에 대해서 CNN inference를 하는 문제가 있었다. 이 문제를 CNN inference를 전체 이미지에 대하여 1회만 수행하고, 이 피쳐맵을 공유하는 방식으로 해결했다.

그러나 여전히 모델을 학습 시키기 위해선 1) 여러 단계를 거쳐야 했고, 2) Fully Connected Layer 밖에 학습 시키지 못하는 한계점이 있었다. 이에 저자는 다음과 같은 주장을 펼친다.

```
CNN 특징 추출부터 classification, bounding box regression 까지 모두 하나의 모델에서 학습시키자!
```

* 주요 keyword
    * RoI Pooling, Multi Task Loss, finetuning for detection

* 전체 알고리즘
    1. 먼저 전체 이미지를 미리 학습된 CNN을 통과시켜 피쳐맵을 추출합니다.
    2. Selective Search를 통해서 찾은 각각의 RoI에 대하여 RoI Pooling을 진행합니다. 그 결과로 고정된 크기의 feature vector를 얻습니다.
    3. feature vector는 fully connected layer들을 통과한 뒤, 두 개의 브랜치로 나뉘게 됩니다.
    4. 하나의 브랜치는 softmax를 통과하여 해당 RoI가 어떤 물체인지 클래시피케이션 합니다. 더 이상 SVM은 사용되지 않습니다.
    5. 또 하나의 브랜치는 bouding box regression을 통해서 selective search로 찾은 박스의 위치를 조정합니다.

CNN을 한번만 통과시킨 뒤, 그 피쳐맵을 공유하는 것은 이미 SPP Net에서 제안된 방법입니다. 그 이후의 스텝들은 SPPNet이나 R-CNN과 그다지 다르지 않습니다.

`본 논문의 가장 큰 특징은 이들을 스텝별로 쪼개어 학습을 진행하지 않고, end-to-end로 엮었다는데 있습니다. 그리고 그 결과로 학습 속도, 인퍼런스 속도, 정확도 모두를 향상시켰다는데 의의가 있습니다.`

### 중요 Key Point

* RoI Pooling
    * Fast R-CNN에서 먼저 입력 이미지는 CNN을 통과하여 피쳐맵을 추출합니다. 추출된 피쳐맵을 미리 정해놓은 H x W 크기에 맞게끔 그리드를 설정합니다. 그리고 각각의 칸 별로 가장 큰 값을 추출하는 max pooling을 실시하면 결과값은 항상 H x W 크기의 피쳐 맵이 되고, 이를 쫙 펼쳐서 feature vector를 추출하게 됩니다. 이러한 RoI Pooling은 앞서 살펴보았던 Spatial Pyramid Pooling에서 피라미드 레벨이 1인 경우와 동일합니다.

    * ![RoI Pooling](./image/RoI_Pooling.png)
