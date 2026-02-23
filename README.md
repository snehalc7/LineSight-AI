# 🎾 LineSight AI

AI-Powered Boundary Verification System

---

## 📌 Overview

LineSight AI is a computer vision-based boundary verification system
designed to determine whether a tennis ball is IN or OUT relative to
a court boundary line using image processing techniques.

This system replicates core principles used in Hawk-Eye systems
through geometric modeling and regression-based boundary detection.

---

## 🧠 How It Works

### 1️⃣ Ball Detection
- HSV color segmentation isolates yellow objects
- Contour filtering selects best candidate
- Circularity validation ensures geometric integrity

### 2️⃣ Boundary Detection
- White strip segmentation via HSV thresholding
- Row-by-row edge extraction
- Robust regression using Huber loss

### 3️⃣ Decision Logic
- Calculates projected boundary at ball height
- Determines IN or OUT via geometric comparison
- Computes margin distance from line

### 4️⃣ Confidence Scoring
Confidence is derived from:

- Base System Score (20 pts)
- Ball Shape Integrity (40 pts)
- Margin Strength (40 pts)

---

## 📊 Decision Classification

Depending on distance from boundary:

- CLEAR IN
- CLOSE CALL
- TOUCHING
- CLOSE OUT
- CLEAR OUT

---

## 🖥️ Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
