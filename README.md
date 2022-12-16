# Improved-Regularization-of-Convolutional-Neural-Networks

Deep learning has been increasingly used in recent years, leading to state-of-the-art performance on different tasks, including object detection [1], semantic segmentation [2], image captioning [3], etc. Most of this success can be attributed to the use of convolutional neural networks (CNNs) [4] that are capable of learning complex hierarchical feature representation of images. The complexity of the tasks given to the deep neural network has increased over time, leading to advanced architectures that require a high computational complexity. These models usually contain tens to hundreds of millions of parameters that are needed for solving a task. However, the greater number of parameters in a network would lead to a higher chance of overfitting, which would then lead to a weak generalizability.

In order to combat overfitting during the training process, researchers have introduced regularization techniques. Regularization techniques incorporate changes into the model in order to reduce the generalization error without harming the training error. A performance improvement can be expected if the regularization technique is chosen carefully. Commonly regularization techniques include dataset augmentation, layer dropout, and weight penalty regularization (L1- or L2-based). Each technique uses a different approach to increase generalizability of the model. Recently, several novel regularization techniques have been introduced to further combat the overfitting issue in large-sized models. In this study, we aim to evaluate the effectiveness of three novel augmentation techniques, namely cutout regularization [5], mixup regularization [6], and self-supervised rotation predictor [7], [8]. The following section discusses the related work.

<img src="https://github.com/fitushar/Improved-Regularization-of-Convolutional-Neural-Networks/blob/main/poster_pic.png"  width="100%" height="100%">



# Part of this codes were adopted from original Implementation of 

* Corrupted CIFAR10: https://github.com/tanimutomo/cifar10-c-eval 
* Mixup: https://github.com/facebookresearch/mixup-cifar10
* Auxialiary rotation loss: https://github.com/hendrycks/ss-ood


# References


* [1]	A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” Adv. Neural Inf. Process. Syst., pp. 1097--1105, 2012.

* [2]	E. Shelhamer, J. Long, and T. Darrell, “Fully Convolutional Networks for Semantic Segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 4, pp. 640–651, 2017.
* [3]	O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, “Show and tell: A neural image caption generator,” Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit., vol. 07-12-June, pp. 3156–3164, 2015.
* [4]	Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278–2323, 1998.
* [5]	T. DeVries and G. W. Taylor, “Improved Regularization of Convolutional Neural Networks with Cutout,” 2017.
* [6]	H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “MixUp: Beyond empirical risk minimization,” 6th Int. Conf. Learn. Represent. ICLR 2018 - Conf. Track Proc., pp. 1–13, 2018.
* [7]	D. Hendrycks, M. Mazeika, S. Kadavath, and D. Song, “Using self-supervised learning can improve model robustness and uncertainty,” Adv. Neural Inf. Process. Syst., vol. 32, no. NeurIPS, 2019.
* [8]	S. Gidaris, P. Singh, and N. Komodakis, “Unsupervised representation learning by predicting image rotations,” 6th Int. Conf. Learn. Represent. ICLR 2018 - Conf. Track Proc., no. 2016, pp. 1–16, 2018.
* [9]	K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.
* [10]	A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” 2009.
* [11]	D. Hendrycks and T. Dietterich, “Benchmarking Neural Network Robustness,” Int. Conf. Learn. Represent., pp. 1–16, 2019.
