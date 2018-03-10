# Arabic-Sentiment-Analysis
A Sentiment-Analysis project that aims at determining the opinion of the user utterance
if it is positive or negative. The project is written in Python language and requires
the Natural Language ToolKit (NLTK) to be installed before usage.

<br />

# System Analysis Stages
Three main stages:

### Pre-processing
Normalization, stop-words removal and stemming.

### Feature Extraction
The Term-Frequency-Inverse-Term-Frequency (IT-IDF) was used as a feature vector for the
classifier

### Classification - Machine Learning
The Support Vector Machine (SVM) and Naive Bias classifiers were used in the experiment.

<br />

# Dataset
Six datasets were used for the experiment

|Data-set                     | Positive Samples  | Negative Samples|
|-----------------------------|-------------------|-----------------|
|Twitter tweets               | (1000) 50%        | (1000) 50%      |
|Product Attraction reviews   | (2073) 96%        | (82) 4%         |
|Hotel Reviews                | (10775) 69%       | (4798) 31%      |
|Movies Reviews               | (969) 63%         | (556) 37%       |
|Product Reviews              | (3101) 72%        | (1172) 28%      |
|Restaurants Reviews          | (8030) 73%        | (2941) 27%      |
|Unified data-set             | (25948) 71%       | (10549) 29%     |

<br />

# Results
The following tables shows the results obtained by the experiment:

SVM classifier results:

|Data-set                     | Precision   | Recall    | F-measure     | Accuracy    |
|-----------------------------|-------------|-----------|---------------|-------------|
|Twitter tweets               |     88%     |    77%    |       81%     |     82%     |
|Product Attraction reviews   |     99%     |    100%   |       98%     |     96%     |
|Hotel Reviews                |     96%     |    98%    |       88%     |     83%     |
|Movies Reviews               |     87%     |    95%    |       82%     |     73%     |
|Product Reviews              |     90%     |    98%    |       86%     |     78%     |
|Restaurants Reviews          |     94%     |    99%    |       85%     |     75%     |
|Unified data-set             |     93%     |    99%    |       83%     |     72%     |

Naive Bias classifier results:

|Data-set                     | Precision   | Recall    | F-measure     | Accuracy    |
|-----------------------------|-------------|-----------|---------------|-------------|
|Twitter tweets               |     88%     |    86%    |       84%     |     84%     |
|Product Attraction reviews   |     88%     |    100%   |       98%     |     96%     |
|Hotel Reviews                |     88%     |    99%    |       83%     |     72%     |
|Movies Reviews               |     88%     |    100%   |       77%     |     63%     |
|Product Reviews              |     88%     |    99%    |       85%     |     74%     |
|Restaurants Reviews          |     94%     |    99%    |       84%     |     73%     |
|Unified data-set             |     92%     |    99%    |       83%     |     72%     |
