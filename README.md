# 🧠 Suicide Risk Detector – NLP Text Classification App

This project is a **web application** that uses **Natural Language Processing (NLP)** and **Machine Learning** to detect potential suicide risk in text. It's designed as a simple and intuitive interface where users can input any message, and the model will classify whether it shows signs of suicidal ideation.

## 🚀 Features

- 🔍 **Text classification model** trained to identify suicide risk.
- ⚙️ **Preprocessing pipeline**: tokenization, stopword removal, lemmatization, etc.
- 💻 **Flask web app** with a dark UI for better readability and visual impact.
- 📊 Model built using **Logistic Regression / XGBoost**.
- 🧪 Includes performance metrics: Confusion Matrix, Accuracy, F1-Score, etc.
- 💾 Model and vectorizer are saved and loaded for real-time predictions.
- 📦 Packaged in a virtual environment for easy reproducibility.

---

## 📁 Project Structure

.
├── app_dashboard.py
├── app_model.py
├── Data
│   ├── cleaned_suicide_detection.parquet
│   ├── data_dashboard.parquet
│   ├── preprocessing_data.parquet
│   └── Suicide_Detection.csv
├── LICENSE
├── Notebooks
│   ├── 01_exploratory_data.ipynb
│   ├── 02_prepocessing.ipynb
│   └── 03_models.ipynb
├── Outputs
│   ├── Images
│   │   ├── avg_token_count_per_class.png
│   │   ├── bigrams_non_suicide.png
│   │   ├── bigrams_suicide.png
│   │   ├── class_distribution.png
│   │   ├── confusion_matrix.png
│   │   ├── general_wordcloud.png
│   │   ├── non_suicide_wordcloud.png
│   │   ├── roc_curve.png
│   │   ├── suicide_wordcloud.png
│   │   ├── top15_char_len.png
│   │   ├── top15_unique_token_count.png
│   │   ├── unigrams_non_suicide.png
│   │   └── unigrams_suicide.png
│   ├── Models
│   │   ├── suicide_detection_model.pkl
│   │   └── Vectorizer.model
│   └── Tables
│       └── quality_inspection.csv
├── __pycache__
│   └── app_model.cpython-310.pyc
├── README.md
├── src
│   ├── Data
│   │   ├── non_suicide_1gram.pkl
│   │   ├── non_suicide_2gram.pkl
│   │   ├── non_suicide_3gram.pkl
│   │   ├── suicide_1gram.pkl
│   │   ├── suicide_2gram.pkl
│   │   └── suicide_3gram.pkl
│   ├── Images
│   │   ├── brain.png
│   │   ├── broken-heart.png
│   │   └── mental-health.png
│   ├── __pycache__
│   │   └── utils.cpython-310.pyc
│   └── utils.py
└── templates
    └── index.html

## 🧠 Model & Data

- Model type: `XGBoostClassifier`
- Vectorization: `TfidfVectorizer`
- Input: Raw text data labeled as `suicidal` or `non-suicidal`
- Dataset: Includes social media posts, forum entries, or public datasets related to mental health.

## 📌 Dependencies

- Flask
- scikit-learn
- nltk
- xgboost
- pandas, numpy, matplotlib, seaborn