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
* `Region Proposal`
    * 주어진 이미지에서 물체가 있을법한 위치를 찾는 것
    * `Selective Search`
        * R-CNN은 룰 베이스 알고리즘을 적용해 `2천개`의 물체가 있을법한 박스를 찾습니다.
        * Selective Search는 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을 법한 박스를 추론하게 됩니다.
        * R-CNN의 Region Proposal 과정 역시 뉴럴 네트워크가 수행하도록 발전했습니다.
* Feature Extraction
    * Selective Search를 통해서 찾아낸 2천개의 박스 영역은 `227x227 크기로 리사이즈` 됩니다.(warp)
    * 그리고 Image Classification으로 `미리 학습되어 있는 CNN 모델`을 통과하여 `4096 size의 특징 벡터를 추출`합니다.
    * 미리 학습되어 있는 모델이란 저자들이 `이미지넷 데이터(ILSVRC2012 classification)`로 미리 학습된 CNN 모델을 가져온 다음, `fine tune`한 것입니다.
    * fine tune을 할 때 실제 Object Detection을 적용할 데이터 셋에서 `ground truth`에 해당하는 이미지들을 가져와 학습시켰습니다.
    * 그리고 Classification의 마지막 레이어를 Object Detection의 클래ㅐ스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로 맞췄습니다.
* 정확도 측정 지표
    * `mAP` 사용
* `정리` : 미리 이미지 넷으로 학습된 CNN을 가져와서, Object Detection 용 데이터 셋으로 fine tuning 한 뒤, selective search 결과로 뽑힌 이미지들로부터 특징 벡터를 추출합니다.
* 