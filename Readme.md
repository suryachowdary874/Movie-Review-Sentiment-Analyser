# Sentiment Analysis using Machine Learning

## Overview

This project performs Sentiment Analysis on movie reviews using Natural Language Processing (NLP) and Machine Learning.

The model analyzes movie review text and predicts whether the sentiment is:

- Positive
- Negative

The project demonstrates the complete NLP workflow including:
- Text preprocessing
- Feature extraction
- Model training
- Prediction
- Model evaluation
- Model saving using Pickle

---

# Features

- Cleans and preprocesses text data
- Converts text into numerical vectors using CountVectorizer
- Trains a Multinomial Naive Bayes model
- Predicts sentiment for unseen reviews
- Evaluates model accuracy
- Saves trained model and vectorizer for future use

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- NLP (Natural Language Processing)
- CountVectorizer
- Multinomial Naive Bayes
- Pickle
- Regular Expressions (re)

---

# Machine Learning Workflow

```text
Movie Reviews
      ↓
Text Cleaning
      ↓
Train-Test Split
      ↓
CountVectorizer
      ↓
Numerical Feature Vectors
      ↓
Multinomial Naive Bayes
      ↓
Prediction
      ↓
Accuracy Evaluation
```

---

# Dataset

Dataset used:

```text
cleaned_imdb_dataset.csv
```

Important Columns:

| Column Name | Description |
|---|---|
| cleaned_review | Preprocessed movie review text |
| sentiment | Review sentiment (Positive / Negative) |

---

# Concepts Used

## 1. CountVectorizer

`CountVectorizer` converts text into numerical vectors by counting the occurrence of words.

Example:

```text
"I love python"
```

becomes:

```text
[1, 1, 1]
```

This allows machine learning models to process text data.

---

## 2. Multinomial Naive Bayes

A probabilistic machine learning algorithm commonly used for:

- Sentiment Analysis
- Spam Detection
- Text Classification

It works well for text data because it uses word frequency information.

---

## 3. Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

This helps evaluate model performance on unseen data.

---

## 4. fit(), transform(), fit_transform()

| Method | Purpose |
|---|---|
| fit() | Learns patterns from training data |
| transform() | Applies learned patterns |
| fit_transform() | Performs both operations together |

Training Data:

```python
vectorizer.fit_transform(X_train)
```

Testing Data:

```python
vectorizer.transform(X_test)
```

---

# Project Structure

```text
Sentiment-Analysis-Project/
│
├── sentiment_analysis.py
├── cleaned_imdb_dataset.csv
├── sentiment_model.pkl
├── README.md
├── requirements.txt
├── notebooks/
└── screenshots/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-project.git
```

## Move into Project Folder

```bash
cd sentiment-analysis-project
```

## Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# How to Run

```bash
python sentiment_analysis.py
```

---

# Sample Prediction

## Input

```text
This movie was absolutely fantastic!
```

## Output

```text
Positive
```

---

# Model Evaluation

Model performance is evaluated using:

```python
accuracy_score(y_test, predictions)
```

The project checks how accurately the model predicts sentiment on unseen reviews.

---

# Saving the Model

The trained model and vectorizer are saved using Pickle:

```python
pickle.dump(artifacts, f)
```

Saved File:

```text
sentiment_model.pkl
```

This avoids retraining the model every time.

---

# Key Learnings

Through this project, I learned:

- NLP workflow
- Text preprocessing
- Feature extraction
- CountVectorizer
- Multinomial Naive Bayes
- Train-test splitting
- fit vs transform
- Model evaluation
- Saving ML models using Pickle
- Handling feature mismatch errors
- Debugging ML pipelines

---

# Future Improvements

- Use TF-IDF Vectorizer
- Try Logistic Regression
- Try Random Forest
- Hyperparameter tuning
- Deploy using Flask or Streamlit
- Add web interface
- Improve preprocessing techniques

---

# Author

Surya Chowdary
