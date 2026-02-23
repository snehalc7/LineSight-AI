# 🎾 LineSight AI  
### AI-Powered Boundary Verification System

---

## 📌 Overview

LineSight AI is a computer vision–based boundary verification system designed to determine whether a tennis ball is **IN or OUT** relative to a court boundary line.

The system uses geometric modeling, HSV color segmentation, and robust regression to replicate core principles of professional line-calling systems in a lightweight and cost-effective way.

---

## 🚀 Features

- 🎯 Accurate IN/OUT decision detection  
- 📏 Distance-based decision classification (Touching, Close Call, Clear In/Out)  
- 📊 Transparent confidence scoring system  
- 🧠 Ball integrity analysis using geometric validation  
- 💻 Interactive Streamlit dashboard UI  
- 🔍 Debug view for mask visualization  

---

## 🧠 How It Works

### 1️⃣ Ball Detection
- HSV color segmentation isolates yellow regions  
- Contour filtering selects best candidate  
- Circularity validation ensures geometric integrity  

### 2️⃣ Boundary Detection
- White strip segmentation via HSV thresholding  
- Row-by-row edge extraction  
- Robust line fitting using Huber regression  

### 3️⃣ Decision Logic
- Projects boundary line at ball height  
- Computes geometric comparison for IN/OUT  
- Calculates exact distance from boundary  

### 4️⃣ Confidence Scoring
Confidence is calculated using:

- Base System Score (20 pts)  
- Ball Shape Integrity (40 pts)  
- Margin Strength from Line (40 pts)  

---

## 📊 Decision Classification

Based on distance from the boundary:

- **CLEAR IN**  
- **CLOSE CALL**  
- **TOUCHING**  
- **CLOSE OUT**  
- **CLEAR OUT**

---

## 🖥️ Running the Project

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
