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
