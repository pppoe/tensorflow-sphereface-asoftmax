## About

This is quick re-implementation of asoftmax loss proposed in this paper: [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063). Please cite it if it helps in your paper.

## Details

1. I was using Tensorflow 1.4
1. I followed this author's caffe implementation [sphereface](https://github.com/wy1iu/sphereface).
1. l is \lambda in the paper to balance the modified logits and original logits

## Visualization of MNIST results

Set l = 1

- original softmax, 97.6758%
![original softmax](https://github.com/pppoe/tensorflow-sphereface-asoftmax/blob/master/figures/m0.png)

- m = 1, 98.0469%
![m = 1](https://github.com/pppoe/tensorflow-sphereface-asoftmax/blob/master/figures/m1.png)

- m = 2, 98.3887%
![m = 2](https://github.com/pppoe/tensorflow-sphereface-asoftmax/blob/master/figures/m2.png)

- m = 4, 98.6523% 
![m = 4](https://github.com/pppoe/tensorflow-sphereface-asoftmax/blob/master/figures/m4.png)

## On Face Recognition 

My observation is that the same set of hyper-parameters does not work well in TF. The asoftmax generally improves the accuracy for about 2% on LFW when trained with CASIA.
The best accuracy I got is about 98.X%. It seems it is quite tricky to tune the hyper-parameters to match the accuracy of the implementation in caffe.
