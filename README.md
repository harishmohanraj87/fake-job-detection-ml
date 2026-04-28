# 🚀 Fake Job Detection System (ML + NLP)

## 🔍 Overview

This project is a **Machine Learning-based Fake Job Detection System** that identifies fraudulent job postings using **Natural Language Processing (NLP)** techniques.
It analyzes job descriptions, filters suspicious patterns, and classifies postings as **Real ✅ or Fake ❌**.

---

## 🎯 Objective

With the rise of online job platforms, fake job listings have become common.
This project aims to:

* Detect fraudulent job postings
* Protect users from scams
* Demonstrate real-world ML + cybersecurity application

---

## ⚙️ Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
* **ML Model:** Logistic Regression / Naive Bayes
* **NLP:** TF-IDF Vectorization
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **Interface (optional):** Streamlit

---

## 🧠 How It Works (Pipeline)

1. **Data Collection**

   * Dataset: `fake_job_postings.csv`

2. **Data Preprocessing**

   * Remove null values
   * Clean text (lowercase, remove punctuation)
   * Remove stopwords

3. **Feature Extraction**

   * Convert text → numerical using **TF-IDF**

4. **Model Training**

   * Train ML model to classify jobs

5. **Prediction**

   * Input job → Output:

     * **Fake ❌**
     * **Real ✅**

---

## 📊 Example

### Input:

```
"Earn $5000 weekly from home, no experience needed"
```

### Output:

```
Prediction: Fake Job ❌
```

---

### Input:

```
"Software Engineer required with Python and 2+ years experience"
```

### Output:

```
Prediction: Real Job ✅
```

---

## 📁 Project Structure

```
fake-job-detection-ml/
│
├── app.py
├── fake_job_internship_detection_system.py
├── fake_job_postings.csv
├── model.pkl
├── tfidf.pkl
├── requirements.txt
├── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/fake-job-detection-ml.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the project:

```
python app.py
```

---

## 📈 Visualizations

* Fake vs Real Job Distribution
* Description Length Analysis
* WordCloud for Fake Job Patterns

---

## 🔐 Cybersecurity Relevance

This project aligns with cybersecurity by:

* Detecting **social engineering attacks**
* Identifying **fraud patterns in job listings**
* Applying **data-driven threat detection**

---

## 🚀 Future Improvements

* Deploy using Streamlit Web App
* Add Deep Learning (LSTM / BERT)
* API integration for job platforms
* Real-time fraud detection

---

## 👨‍💻 Author

**Harish Mohanraj**
B.Tech CSE (Cybersecurity)
D.Y Patil International University

---

## ⭐ If you found this useful

Give this repo a star ⭐ and feel free to contribute!
