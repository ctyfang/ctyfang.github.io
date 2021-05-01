---
layout: post
title: Primer on 3D Object Detection
description: PointNet, PointNet++, and PointPillar
image: assets/images/pointnet-banner.jpg
---

## Intended Audience

An effective way to verify that you understand something is to try and teach it. As I learn more about 3D object detection through these seminal papers, I hope to reinforce my own learning by writing these posts. For anybody on a similar journey, I hope to provide some clarity on the papers covered. The intended audience is anybody with a bit of background knowledge in robotics.

## Background: What is 3D Object Detection?

For robots to affect their environment, a 3D understanding is needed. LiDARs are sensing devices that help to solve this problem. LiDARs generate a "cloud" of 3D points (aka a "pointcloud") describing the structure of the environment. By processing these pointclouds, we can detect 3D objects within them, classify them according to some categories (eg pedestrians, cyclists, etc), and enable our robots to plan a path for movement in 3D. A basic approach for detecting objects in pointclouds could involve clustering with an approach like K-Means. There are several issues with this naive approach. Firstly, K-Means requires us to specify the number of clusters, but the number of objects in each pointcloud is likely to differ. Secondly, because conventional mechanical LiDAR generate pointclouds by emitting laser beams radially, as the radial distance increases, the beams will become more spread-out leading to greater sparsity. Presently, this detection is performed with deep neural networks. 

<img align="middle" src="../../../assets/images/pointnet-banner.jpg" alt="drawing" width="80%"/>

## Introduction to PointNets

There are several ways to represent 3D data: voxels, pointclouds, and meshes are some examples. Pointclouds are the raw representation from the sensor and are difficult to work with due to their unordered nature. Voxels are a more structured representation akin to images involving a discretized 3D occupancy grid. While they are more structured, they are extremely memory inefficient as large areas of the 3D grid will be empty. PointNets were a revolutionary architecture that allowed us to learn tasks directly on pointclouds, leveraging their memory efficiency. While PointNets could not perform object detection on un-segmented pointclouds, they were an important stepping-stone.

<img align="middle" src="../../../assets/images/pointnet-overview.jpg" alt="drawing" width="80%"/>
<img align="middle" src="../../../assets/images/pointnet-architecture.jpg" alt="drawing" width="80%"/>

Focusing on the classification task, the dataset consists of segmented pointclouds (each input pointcloud only contains points that belong to the object) and ground-truth class labels. For the nominal network, pointclouds with 1024 points are used, but this can be generalized to N points. As shown in the architecture diagram, the input to the network is an Nx3 array of XYZ coordinate points. In practice, this can be generalized to NxC for C channels if including point properties such as color and reflection intensity. Ignoring the T-Net modules for now, each point in the input array is processed by a two-layer neural network indicated by MLP (multi-layer perceptron) in the diagram. In this way, each 3-vector in the array is mapped to a 64-vector. Points are further processed by a second MLP and mapped to 1024-vectors. Across the 1024 dimensions, a max-pool operation is performed, resulting in a single 1024-vector that describes the entire pointcloud. This global description vector is fed into a final MLP to produce the object type classification.

Before discussing the T-Nets, we first discuss the difficulties in designing a pointcloud-based network: we seek invariance to geometric transformations (rotation and translation), and ordering. Given a pointcloud for a chair, for instance, the network should still predict the chair class if we flipped the chair upside-down, or shuffled the points in the array. T-Nets are aimed at the geometric invariance. Each T-Net predicts a corrective transformation that re-orients the point set to a canonical orientation - for instance we may always want a chair to be right-side up. The invariance to ordering is achieved through the max pooling operation and shared MLP weights. Regardless of input ordering, after the max pooling operation across the 1024 feature dimensions, we will end up with the same global descriptor (and subsequently the same class prediction).

Compared to other state-of-the-art networks at the time for classification on 3D input, PointNet achieved the best overall accuracy, despite only needing one view (the majority of the other SoTA networks used 10+ views). In addition to high classification accuracy, the authors showed that the performance was robust to missing points. They posit that this robustness comes from the network's ability to represent objects using a small subset of the points (critical points). The critical points are those which contribute to the global description vector via the max pooling operation.

<img align="middle" src="../../../assets/images/pointnet-degrad.jpg" alt="drawing" width="80%"/>
<img align="middle" src="../../../assets/images/pointnet-critical.jpg" alt="drawing" width="80%"/>

## An Improvement: PointNet++

<img align="middle" src="../../../assets/images/pointnet++-architecture.jpg" alt="drawing" width="80%"/>

While PointNet worked well, it did not leverage the local structure of different regions in the pointcloud. PointNet++ uses similar operating principles as CNN's, where neurons with progressively larger receptive fields extract features with progressively larger scale and complexity. We focus on the hierarchical feature learning module first which is defined by repeated "set abstraction". Given the raw pointcloud, we subsample N1 points by repeatedly taking the farthest (by some distance metric) point from the current subsampled set. Each of these N1 subsampled points serves as the center of a spherical neighborhood from which local structor will be encoded. A feature vector for each of the N1 neighborhoods is generated by processing the K nearest points through a small PointNet. We are left with N1 feature descriptors for each of the subsampled points. We can repeat this process of subsampling and neighborhood feature encoding to generate a feature hierarchy. This feature hierarchy can then be used for tasks like semantic segmentation and classification. In the paper, both tasks are demonstrated.

<img align="middle" src="../../../assets/images/pointnet++-classification.jpg" alt="drawing" width="50%"/>
<img align="middle" src="../../../assets/images/pointnet++-semantic.jpg" alt="drawing" width="50%"/>

While the features from the last layer (with the largest receptive field) can simply be used for classification, semantic segmentation requires more fine-grained information to classify each individual point. The segmentation head of PointNet++ takes the point features from the last encoding layer and gradually restores the spatial resolution, incorporating features from earlier encoding layers along the way. Given the N2 features from the last encoding layer, we interpolate the feature vectors for the N1 points in the second last encoding layer. These N1 features are concatenated with the N1 features from the earlier encoding layer, and the concatenated vector is processed with a PointNet. This process of interpolation, concatenation, and processing via MLP is the same principle as U-Net's use of upsampling, concatenation, and combination via convolution.

## Recent Improvements: PointPillars
