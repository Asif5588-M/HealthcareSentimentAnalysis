---
title: Healthcare Sentiment Analysis
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# 🏥 Healthcare Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![TextBlob](https://img.shields.io/badge/TextBlob-NLP-orange)

## 📋 Problem Statement
Healthcare organizations receive thousands of patient reviews daily. Manually analyzing this feedback is time-consuming and inconsistent. This AI system automatically detects sentiment to help hospitals improve patient experience.

## ✨ Features
- 📝 Single review analysis
- 📊 Bulk CSV upload & analysis
- 📈 Interactive dashboard with charts
- ☁️ Word Cloud visualization
- 🕐 Analysis history from MongoDB
- 📥 Download results as CSV

## 🎯 Use Cases
- Hospital patient review analysis
- Drug review sentiment tracking
- Healthcare app feedback analysis
- Social media health discussion monitoring

## 🛠️ Tech Stack
- **NLP:** TextBlob
- **Database:** MongoDB Atlas
- **Dashboard:** Streamlit
- **Charts:** Plotly
- **Visualization:** WordCloud

## 🌐 Live Demo
[healthcare-sentiment-asif.streamlit.app](https://healthcare-sentiment-asif.streamlit.app)

## 🚀 Run Locally
```bash
git clone https://github.com/Asif5588-M/HealthcareSentimentAnalysis.git
cd HealthcareSentimentAnalysis
conda create -n sentiment python=3.10 -y
conda activate sentiment
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Author
**Asif Nawaz** — Data Scientist | Healthcare Analytics
- GitHub: [@Asif5588-M](https://github.com/Asif5588-M)
- Live: [healthcare-sentiment-asif.streamlit.app](https://healthcare-sentiment-asif.streamlit.app)