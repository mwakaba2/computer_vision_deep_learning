### Question 6: Deep Learning is King
Explain in your own words why Deep Learning approaches has become the state of art approach in solving traditional CV and PR problems. Why does image recognition and object detection work the best using neural networks as opposed to some of the other older traditional CV approaches?


### Answer

Deep learning has become the state of the art approach for several reasons. First, given more data, it can learn and generalize more accurately than traditional ml methods. Second, it creates features from the input data, and this eliminates the tedious task of feature engineering. Finally, we can use pretrained models and transfer learn to solve related tasks quickly and with less data.

Before deep learning, image recognition and object detection were solved by handcrafting the features. For example, it was common to use descriptors such as SIFT and HOG features. This would then be fed into traditional ML classifiers (e.g. SVM, Random Forest). Finding the right set of features and classifier for one image dataset could take many tries. Once the right set of features and classifier was found, the same techniques could not be applied to another task.

Convolutional neural networks handle feature engineering. CNN can learn the features from the training data directly. The more data it trains on, the more patterns and generalizations it can find. In the earlier layers, it learns very low level features such as edges and lines. In the later layers, it learns to recognize patterns, objects, and people. Because of this, a CNN that's been trained with multiple GPUs on a very large dataset (e.g. ImageNet or Pascal VOC dataset) can be used as a base to solve tasks with different image datasets. For example, specific dog breeds can be recognized very quickly by updating the pretrained model's last fully connected layer to recognize only dog breeds and fine-tuning. Thanks to transfer learning, we can re-use the models and get good results with less data and computing resources.

