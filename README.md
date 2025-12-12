# 📰 Fake News Detection using Machine Learning

Detecting misinformation using NLP and machine learning techniques

## 📌 Overview

Fake news detection is an important task in modern digital environments, where misinformation can spread rapidly.
This project builds a **Fake News Classification Model** using **Natural Language Processing (NLP)** and **Machine Learning**, classifying news articles as **REAL** or **FAKE**.

The model uses text preprocessing, feature extraction (TF-IDF), and classifiers like **Logistic Regression**, **Naive Bayes**, **SVM**, or deep-learning models (LSTM/BERT optionally).

---

## 🚀 Features

* Clean and well-structured dataset preprocessing
* TF-IDF vectorization for feature extraction
* Multiple ML models for comparison
* Model evaluation using:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix
* Jupyter Notebook for easy experimentation
* Ready-to-train pipeline

---

## 📁 Project Structure

```
fake-news-detection/
│── data/
│   ├── train.csv
│   ├── test.csv
│── notebooks/
│   ├── fake_news_detection.ipynb
│── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── utils.py
│── models/
│── README.md
│── requirements.txt
│── app.py  (optional Flask/Streamlit app)
```

---

## 🧠 Workflow

### 1️⃣ Import Dataset

Use Kaggle datasets listed below.

### 2️⃣ Data Cleaning

* Remove punctuation
* Lowercasing
* Stopword removal
* Tokenization
* Lemmatization

### 3️⃣ Feature Engineering

* TF-IDF Vectorizer
* Optional: Word2Vec / BERT embeddings

### 4️⃣ Model Training

Algorithms used:
✔ Logistic Regression
✔ Passive-Aggressive Classifier
✔ Naive Bayes
✔ Random Forest
✔ SVM

### 5️⃣ Evaluation

Use classification metrics & confusion matrix.

---

## 📊 Kaggle Datasets for Fake News Detection (Recommended)

### 🔗 **1. Fake News Dataset**

Best dataset for ML models
👉 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### 🔗 **2. Fake News Prediction Dataset**

Contains labelled True/False articles
👉 [https://www.kaggle.com/datasets/jruvika/fake-news-detection](https://www.kaggle.com/datasets/jruvika/fake-news-detection)

### 🔗 **3. LIAR Dataset (Short statements)**

Short political statements labelled as true/false
👉 [https://www.kaggle.com/datasets/mrisdal/fake-news](https://www.kaggle.com/datasets/mrisdal/fake-news)

### 🔗 **4. News Authenticity Dataset (Balanced)**

Good for binary classification
👉 [https://www.kaggle.com/datasets/saurabhshahane/news-articles-dataset](https://www.kaggle.com/datasets/saurabhshahane/news-articles-dataset)

### 🔗 **5. Fake NewsNet Dataset (Advanced)**

Includes metadata + social engagement
👉 [https://www.kaggle.com/datasets/jruvika/fake-news-detection](https://www.kaggle.com/datasets/jruvika/fake-news-detection)

---

## 🛠️ Technologies & Libraries

* Python
* NumPy
* Pandas
* Scikit-learn
* NLTK / SpaCy
* Matplotlib / Seaborn
* TensorFlow / PyTorch (optional)

---

## ▶️ How to Run

```bash
git clone https://github.com/yourusername/fake-news-detection
cd fake-news-detection
pip install -r requirements.txt
python src/train_model.py
```

---

