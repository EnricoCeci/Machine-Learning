# Face Detection for Digital Cameras

---

## Overview

This project consists of a Computer Vision application developed in Python to detect human faces in digital images and return the corresponding bounding-box coordinates.

The system was designed under specific constraints: it uses **Scikit-learn**, does not rely on pretrained models and is intended to operate with limited computational resources.

The project combines dataset construction, image preprocessing, HOG feature extraction, linear SVM classification, multiscale sliding-window detection, Hard Negative Mining and Non-Maximum Suppression.

The developed solution allows users to:

* build a custom training dataset from positive and negative image samples;
* create a demographically balanced positive face dataset using FairFace;
* generate negative training patches from images without faces;
* increase positive-sample variability through image augmentation;
* extract Histogram of Oriented Gradients (HOG) features from image patches;
* train and optimise a Linear SVM classifier;
* prevent data leakage between related image patches through grouped cross-validation;
* scan complete images using a multiscale sliding-window approach;
* identify difficult false positives through Hard Negative Mining;
* reduce duplicate detections using Non-Maximum Suppression based on Intersection over Union;
* tune the detector decision threshold on a validation set;
* evaluate final face-detection performance on previously unseen images.

The project was developed following a predefined specification, with particular attention to computational efficiency, model training from scratch and documentation of the methodological choices.

---

## Technologies

* Python 3
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Scikit-image
* XML ElementTree
* KaggleHub
* Google Colab

---

## Project Requirements

### Face Detection System

The objective of the project is to develop a face-detection pipeline capable of identifying faces in digital images and returning their bounding-box coordinates.

The system must:

* accept an image as input;
* detect whether one or more faces are present;
* return the coordinates of the detected bounding boxes;
* return an empty list when no face is detected;
* use Scikit-learn for model development;
* train the model from scratch without pretrained detectors;
* operate under limited computational resources.

The project also requires:

* identification or construction of suitable datasets;
* research into established Computer Vision approaches;
* documentation of preprocessing, modelling and optimisation choices;
* references to external academic and technical resources used during development.

---

## Datasets

The project uses different datasets for classifier training and detector evaluation.

### FairFace

The **FairFace** dataset is used as the source of positive face patches for classifier training.

It was selected because it provides a broad demographic representation across:

* age;
* gender;
* ethnicity.

A stratified sample was created by considering combinations of these three variables.

The dataset contains:

* 9 age groups;
* 2 gender groups;
* 7 ethnicity groups.

This results in **126 possible demographic combinations**.

Three images were sampled from each combination, producing:

* **378 original positive face patches**.

The patches were subsequently resized to a fixed size of:

* **145 × 100 pixels**.

Positive augmentation was then applied, doubling the positive training population from:

* **378 original positive patches**
* to **756 total positive patches**.

FairFace resources:

[FairFace GitHub Repository](https://github.com/joojs/fairface)

[FairFace Paper](https://arxiv.org/pdf/1908.04913)

---

### Negative Images

Negative samples were obtained from images without human faces.

Multiple random patches were extracted from these images and resized to the same dimensions used for the positive samples.

These patches allow the classifier to learn the distinction between facial HOG patterns and non-facial visual structures.

Additional negative images unseen during initial training were later used for **Hard Negative Mining**.

---

### Kaggle Face Detection Dataset

The final detector was validated and tested using the Kaggle:

**Face Detection - Face Recognition Dataset**.

The dataset contains:

* **100 selfie images**;
* **100 different individuals**;
* bounding-box annotations stored in an XML file.

Unlike FairFace, these are complete photographs rather than tightly cropped face patches and therefore provide a more realistic environment for evaluating the detector.

The dataset was split by image into:

* **50% validation set**;
* **50% test set**.

The annotated bounding boxes were also analysed to estimate typical face dimensions and guide the selection of:

* training patch size;
* image resizing parameters;
* sliding-window scales;
* sliding-window step size.

Dataset:

[Face Detection - Face Recognition Dataset](https://www.kaggle.com/datasets/trainingdatapro/face-detection-photos-and-labels)

---

## Methodology

The project includes the following stages.

### Dataset Construction

* Import and parse the Kaggle XML annotations.
* Analyse the dimensions of annotated face bounding boxes.
* Build a stratified sample from FairFace based on:

  * age;
  * gender;
  * ethnicity.
* Resize FairFace face patches to **145 × 100 pixels**.
* Import negative images without faces.
* Extract random negative patches.
* Assign group identifiers to samples originating from the same source image.

---

### Data Augmentation

Moderate augmentation is applied exclusively to positive face patches in order to increase variability while preserving realistic facial structures.

The implemented transformations include:

* rotation;
* translation;
* brightness variation;
* combinations of the above transformations.

Preview functions are used to visually inspect transformed patches before applying augmentation to the complete positive dataset.

The augmentation process doubles the number of positive samples from **378 to 756**.

---

### HOG Feature Extraction

Each image patch is transformed into a numerical representation using **Histogram of Oriented Gradients (HOG)**.

The HOG transformation captures information related to:

* local gradients;
* edges;
* contours;
* orientation patterns.

The process includes:

* computation of image gradients;
* division of the patch into cells;
* creation of orientation histograms;
* block normalisation;
* concatenation into a one-dimensional feature vector.

HOG was selected because it provides an efficient representation of shape and local gradient structures while requiring substantially fewer computational resources than modern deep-learning approaches.

---

### Machine Learning Pipeline

A custom HOG transformer compatible with the Scikit-learn interface is integrated into a `Pipeline`.

The pipeline includes:

* HOG feature extraction;
* feature standardisation;
* `LinearSVC` classification.

Model selection is performed through `GridSearchCV`.

Different values of the SVM regularisation parameter `C` are evaluated.

The optimisation process monitors:

* Accuracy;
* Precision;
* Recall;
* F1-score.

The final model is selected using **F1-score as the refit metric**, providing a balance between recall and precision.

---

### Grouped Cross-Validation

`GroupKFold` is used instead of standard cross-validation.

This is necessary because several samples may originate from the same source image, including:

* multiple negative patches extracted from the same image;
* original positive patches and their augmented versions.

Samples belonging to the same source image are assigned the same group identifier.

This prevents correlated samples from being distributed across both training and validation folds, reducing the risk of data leakage.

---

### Multiscale Sliding Window

The trained patch-level classifier is converted into a complete face detector using a multiscale sliding-window approach.

The detector progressively resizes the input image and applies a fixed-size sliding window at each scale.

The final configuration uses:

* patch size: **145 × 100 pixels**;
* sliding-window step: **12 pixels**;
* scales:

  * `1.0`;
  * `0.75`;
  * `0.5`.

Scaling the complete image rather than resizing each individual candidate patch helps reduce computational cost.

For every extracted window:

1. the classifier computes a decision score;
2. windows below a minimum score are discarded;
3. surviving candidates are converted back to the coordinates of the original image.

---

### Hard Negative Mining

Hard Negative Mining is used to improve the classifier's robustness against difficult false positives.

The procedure consists of:

* applying the detector to complete images containing no faces;
* extracting candidate patches through the multiscale sliding window;
* obtaining classification scores from the trained SVM;
* identifying negative patches receiving relatively high face scores;
* adding selected hard negatives to the training dataset;
* retraining the classification pipeline.

To avoid introducing excessively easy or anomalous samples, hard negatives are selected within the:

* **85th to 95th percentile** of classification scores.

Redundancy is further reduced by retaining a maximum of:

* **3 hard-negative patches per source image**.

After Hard Negative Mining, patch-level cross-validation metrics decrease by approximately **3–5%**.

This reduction is expected because the classifier is exposed to more ambiguous negative examples. The objective of HNM is therefore not to maximise patch-level metrics but to improve robustness during detection on complete images.

---

### Non-Maximum Suppression

A single face can generate multiple highly overlapping detections because adjacent sliding windows may capture the same facial pattern.

To reduce these duplicate bounding boxes, the project implements **Non-Maximum Suppression (NMS)**.

The overlap between two bounding boxes is calculated using:

**Intersection over Union (IoU)**.

The algorithm:

1. sorts bounding boxes according to classification score;
2. keeps the highest-scoring box;
3. compares it with the remaining boxes;
4. removes boxes whose IoU exceeds the selected threshold;
5. repeats the process until all remaining boxes have been evaluated.

The detector uses an NMS IoU threshold of:

* **0.30**.

---

### Detector Validation

The detector decision threshold is selected using the validation dataset.

The following thresholds are evaluated:

* `0.8`
* `1.0`
* `1.1`
* `1.2`
* `1.4`

A threshold of **1.2** provides the most balanced trade-off between positive- and negative-image classification.

However, the final detector uses:

* **decision threshold = 1.1**

because the application is intended primarily for selfie photography.

In this context, missing an actual face is considered more critical than accepting a moderate number of residual false positives.

The slightly more permissive threshold therefore prioritises detector sensitivity.

---

## Project Structure

```text
face-detection/
│
├── .gitignore
├── README.md
└── Face_Detection_Digital_Cameras_EN.ipynb
```

---

## Main Findings

The patch-level classifier initially achieves encouraging classification performance:

* Recall of approximately **0.92**;
* Precision of approximately **0.97**.

These metrics indicate that the HOG + Linear SVM classifier can distinguish isolated positive and negative patches effectively.

However, patch-level performance cannot be directly interpreted as detector performance.

During complete-image detection, the sliding-window procedure evaluates a very large number of candidate regions, including:

* partial faces;
* background textures;
* high-contrast structures;
* objects with face-like gradient patterns.

Consequently, even a relatively small false-positive rate at patch level may generate multiple false detections at image level.

Hard Negative Mining makes the classification problem more difficult, producing an expected decrease of approximately **3–5%** in patch-level cross-validation metrics while exposing the model to more representative difficult negatives.

At detector level:

* the selected validation configuration with decision threshold **1.1** achieves approximately **0.90 recall**;
* on the test set, recall decreases to approximately **0.80**;
* test-set specificity reaches approximately **0.72**.

The difference between validation and test performance indicates some sensitivity to changes in image distribution, which is expected given the relatively small evaluation dataset and the differences between the patch-level training data and complete real-world images.

Qualitative inspection shows that:

* the detector identifies faces successfully in most positive images;
* false positives frequently occur in regions containing strong contrasts;
* regular geometric structures and graphical patterns may be interpreted as face-like structures;
* this behaviour is consistent with the properties of HOG features, which represent local gradient patterns rather than semantic image content.

Overall, the detector provides a reasonable compromise between computational cost and detection capability under the project constraints.

---

## Design Considerations

The system reproduces the classical architecture of pre-deep-learning object detectors.

Its main components are:

* HOG feature extraction;
* Linear SVM classification;
* multiscale image search;
* sliding-window candidate generation;
* Hard Negative Mining;
* Non-Maximum Suppression.

This architecture was chosen because it satisfies the main project constraints:

* no pretrained models;
* model training from scratch;
* Scikit-learn compatibility;
* limited computational resources;
* transparent and modular processing pipeline.

The detector is therefore not intended to compete with modern deep neural network face detectors.

Instead, it demonstrates how a complete face-detection pipeline can be constructed from classical Computer Vision and Machine Learning techniques while maintaining control over the entire training and detection process.

---

## References

The implementation and methodological choices were inspired by the following resources.

### Python Data Science Handbook

Jake VanderPlas' face-detection example provides the methodological foundation for the HOG + SVM and sliding-window architecture.

[Python Data Science Handbook – Application: A Face Detection Pipeline](https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html)

### Histograms of Oriented Gradients

The HOG-based approach is inspired by the reference work by Navneet Dalal and Bill Triggs:

**Histograms of Oriented Gradients for Human Detection**

[Dalal & Triggs Paper](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

### Non-Maximum Suppression

The implementation of Non-Maximum Suppression was inspired by the PyImageSearch article:

[(Faster) Non-Maximum Suppression in Python](https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)

### Intersection over Union

[Intersection over Union for Object Detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

---

## How to Run

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image kagglehub
```

Open the notebook in Jupyter Notebook, JupyterLab or Google Colab:

```text
Face_Detection_Digital_Cameras_EN.ipynb
```

The project uses Google Drive for some image datasets and therefore the corresponding paths must be updated according to the user's directory structure.

The Kaggle validation and test dataset is downloaded automatically through `kagglehub`.

Then run the notebook cells in order.

The notebook performs:

* library import;
* Kaggle dataset download;
* XML annotation parsing;
* validation-test splitting;
* bounding-box dimension analysis;
* FairFace stratified sampling;
* face-patch resizing;
* negative-patch extraction;
* positive-sample augmentation;
* training dataset construction;
* HOG feature extraction;
* Linear SVM optimisation through grouped cross-validation;
* Hard Negative Mining;
* classifier retraining;
* multiscale sliding-window detection;
* Non-Maximum Suppression;
* detector threshold validation;
* final test-set evaluation;
* qualitative visualisation of detected bounding boxes.
