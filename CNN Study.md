# 1. R-CNN

##### 참고 사이트 : https://yeomko.tistory.com/13?category=851298

* 중요 Keyword : Region Proposal, Seletive Search, 2000개의 bounding box, 227x227 size(resize=warp), 4096 차원의 특징 벡터,
Non-Maximum Supperssion, IoU(Intersection over Union) => 기준 0.5, Bounding Box Regression



* 찾아볼 용어 : fine tune, ground truth, mAP(object Detection의 정확도 측정 지표)

* R-CNN Object Detection 알고리즘 수행 절차
    * 1.Input 이미지에 Selective Search 알고리즘을 적용해 물체가 있으란한 박스 2천개를 추출한다.
    * 2.추출한 모든 박스를 227x227 크기로 resize(warp)한다. 이 때 박스의 비율 등은 고려하지 않는다.
    * 3.미리 image-Net 데이터를 통해 학습시켜놓은 CNNㅇ을 통과시켜 4096 차원의 특징 벡터를 추출한다.
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
