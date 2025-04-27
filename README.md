# Person_Recognition using Gait-Matching

### Introduction
Gait Recognition, the process of identifying individuals based on their walking patterns, has emerged as a significant area of research in biometric authentication. Unlike traditional methods, such as fingerprints and facial recognition, gait-based identification is non-invasive, unobtrusive, and efficient even at a distance without the subject’s cooperation. This renders gait an extremely valuable biometric, particularly for security monitoring, surveillance, and medical applications.

### Aim for the project:
In this project, we aim to build a Person Recognition System by extracting discriminative gait features from different images and applying a kurtosis-based feature selection method. Our approach uses a ResNet101 for feature extraction, combined with a Kurtosis Controlled Entropy (KcE) framework to enhance feature discrimination by reducing redundancy and noise. As a result, the proposed system achieved good test accuracy as discussed in section 5 on the CASIA-C dataset, demonstrating baseline performance in challenging real-world conditions. Feature selection through Kurtosis-Controlled Entropy (KcE) effectively reduced input dimensionality, improving training efficiency.

### Datasets 

1) CASIA-B: A widely used extensive dataset used for gait recognition, consisting of video sequences of 124 subjects captured from 11 different angles under 3 conditions (walking with a bag, walking with a coat, and normal walking). However, due to computational constraints, we have used a subset of the CASIA B dataset comprising sequences of fewer subjects with fewer frames.
2) CASIA-C: A widely used benchmark dataset for gait recognition, containing silhouettes captured under different clothing and view variations.
3) Real-Time captured videos: To simulate real-world scenarios, we also employed a real-time video dataset with sequences of two subjects.
