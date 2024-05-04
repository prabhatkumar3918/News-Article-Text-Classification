# News-Article-Text-Classification
This repository contains code for classifying news articles into different categories such as World, Sports, Business, and Science/Tech using machine learning algorithms. The classification is based on the text content of the news articles.

Table of Contents
>Introduction
>Dataset
>Requirements
>Preprocessing
>Training
>Evaluation
>Results
>Contributing

Introduction:

Text classification is the process of categorizing text documents into predefined classes or categories. In this project, we aim to classify news articles into four categories: World, Sports, Business, and Science/Tech. We use various machine learning algorithms for training and evaluation.

Dataset:

The dataset used in this project consists of news articles labeled with their respective categories. The dataset is split into training and test sets, with each news article labeled with its class index.

Requirements
To run the code in this repository, you need the following dependencies:

>Python (>=3.6)
>pandas
>numpy
>seaborn
>nltk
>scikit-learn
>wordcloud
>matplotlib
You can install these dependencies using pip:

Preprocessing:

The preprocessing steps include:

Removing HTML tags, URLs, and special characters
Converting text to lowercase
Tokenization
Removing stopwords
Stemming and Lemmatization

Training:

The training phase involves:

Vectorizing text data using TF-IDF (Term Frequency-Inverse Document Frequency)
Training machine learning models:
Multinomial Naive Bayes
Logistic Regression
Random Forest
Support Vector Classifier
Decision Tree Classifier
K Nearest Neighbors
Gaussian Naive Bayes

Evaluation:
The evaluation metrics used include:

Accuracy
Precision
Recall
F1-score

Results:

The performance of each model is evaluated using the test dataset, and the results are displayed using confusion matrices and various performance metrics.

Contributing:

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.


