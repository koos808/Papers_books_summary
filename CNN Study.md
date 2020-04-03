# 1. R-CNN :
##### 논문 : Rich feature hierarchies for accurate object detection and semantic segmentation
##### 참고 사이트 1 : https://yeomko.tistory.com/13?category=851298
##### 참고 사이트 2 : https://jaehyeongan.github.io/2019/10/10/R-CNN/

* 중요 Keyword : Region Proposal, Seletive Search, 2000개의 bounding box, 227x227 size(resize=warp), 4096 차원의 특징 벡터,
Non-Maximum Supperssion, IoU(Intersection over Union) => 기준 0.5, Bounding Box Regression

* 찾아볼 용어 : fine tune, ground truth, mAP(object Detection의 정확도 측정 지표)

* 알아 두면 좋은 정보
    * 1.우선 `Obejct Detection`이란 이미지가 무엇인지 판단하는 Classification과 이미지 내의 물체의 위치 정보를 찾는 Localization을 수행하는 것을 말한다.

<br/>
<br/>

* R-CNN Object Detection 알고리즘 수행 절차
    * 1.Input 이미지에 Selective Search 알고리즘을 적용해 물체가 있으란한 박스 2천개를 추출한다.
    * 2.추출한 모든 박스를 227x227 크기로 resize(warp)한다. 이 때 박스의 비율 등은 고려하지 않는다.
    * 3.미리 image-Net 데이터를 통해 학습시켜놓은 CNN을 통과시켜 4096 차원의 특징 벡터를 추출한다.
    * 4.추출된 벡터를 가지고 각각의 클래스(Object의 종류)마다 학습시켜놓은 SVM Classifier를 통과시킨다.
    * 5.Bounding Box Regression을 적용해 박스의 위치를 조정한다.

### 중요 Key Point
* 1.`Region Proposal`
    * 주어진 이미지에서 물체가 있을법한 위치를 찾는 것
    * `Selective Search`
        * R-CNN은 룰 베이스 알고리즘을 적용해 `2천개`의 물체가 있을법한 박스를 찾습니다.
        * Selective Search는 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을 법한 박스를 추론하게 됩니다.
        * R-CNN의 Region Proposal 과정 역시 뉴럴 네트워크가 수행하도록 발전했습니다.
* 2.`Feature Extraction`
    * Selective Search를 통해서 찾아낸 2천개의 박스 영역은 `227x227 크기로 리사이즈` 됩니다.(warp)
    * 그리고 Image Classification으로 `미리 학습되어 있는 CNN 모델`을 통과하여 `4096 size의 특징 벡터를 추출`합니다.
    * 미리 학습되어 있는 모델이란 저자들이 `이미지넷 데이터(ILSVRC2012 classification)`로 미리 학습된 CNN 모델을 가져온 다음, `fine tune`한 것입니다.
    * fine tune을 할 때 실제 Object Detection을 적용할 데이터 셋에서 `ground truth`에 해당하는 이미지들을 가져와 학습시켰습니다.
    * 그리고 Classification의 마지막 레이어를 Object Detection의 클래스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로 맞췄습니다.
* 정확도 측정 지표
    * `mAP` 사용
    * `정리` : 미리 이미지 넷으로 학습된 CNN을 가져와서, Object Detection 용 데이터 셋으로 `fine tuning` 한 뒤, `selective search 결과로 뽑힌 이미지들로부터 특징 벡터를 추출`합니다.
* 3.Classification
    * CNN을 통해 추출한 벡터로 각각의 클래스 별로 `SVM Classifier`를 학습시킵니다.
    * 주어진 벡터를 놓고 이것이 해당 물체가 맞는지 아닌지를 구별하는 Classifier 모델을 학습시킵니다. 그런데 왜 CNN Classifier를 놔두고 SVM을 사용할까? -> 저자의 답변 : "그냥 CNN Classifier를 쓰는 것이 SVM을 썼을 때 보다 mAP 성능이 4%가 낮아졌다. 이는 아마도 fine tuning 과정에서 물체의 위치 정보가 유실되고 무작위로 추출된 샘플을 학습하여 발생한 것으로 보입니다."
    * 이후 R-CNN 에서는 SVM을 붙혀서 학습시키는 기법이 더 이상 사용되지 않는다.
* 4.`Non-Maximum Suppression`
    * Classification 단계에서 SVM을 통과하여 `각각 박스들은 어떤 물체일 확률 값(Score)`을 가지게 되었습니다. -> 여기서 의문점 : `2천개 박스가 모두 필요한 것인가?` -> 물론 `NO NO`
    * 동일한 물체에 여러 개 박스가 쳐져 있다면, 가장 스코어가 높은 박스만 남기고 나머지는 제거하고, 이 과정을 Non-Maximum Suppression이라 합니다.
    * 이 때 서로 다른 두 박스가 동일한 물체에 쳐져 있다고 `어떻게 판별`할 수 있을까요? -> 여기서 `IoU(Intersection over Union) 개념`이 적용됩니다. 쉽게 말하면 두 박스의 `교집합/합집합 값`을 의미하며 두 박스가 일치할 수록 1에 가까운 값에 나오게 됩니다.
    * 논문에서는 Iou가 `0.5보다 크면` 동일한 물체를 대상으로 한 박스로 판단하고 Non-Maximum Suppression을 적용합니다.
* 5.`Bounding Box Regression`
    * 위의 단계를 거치며 물체가 있을 법한 위치를 찾고, 해당 물체의 종류를 판별할 수 있는 Classification Model을 학습시켰습니다. 하지만 위의 단계까지만 거치면 Selective Search를 통해서 찾은 박스 위치는 상당히 부정확하게 됩니다.
    * 따라서 성능을 향상시키기 위해 `박스 위치를 조정해주는 단계`를 거치는데 이를 `Bounding Box Regression`이라 합니다. 
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
    * 속도 저하의 가장 큰 병목 구간은 selective search를 통해서 찾은 2천개의 영역에 모두 CNN inference를 진행하기 때문입니다.

* R-CNN에서 학습이 일어나는 부분
    * 1.이미지 넷으로 이미 학습된 모델을 가져와 fine tuning 하는 부분
    * 2.SVM Classifier를 학습시키는 부분
    * 3.Bounding Box Regression


