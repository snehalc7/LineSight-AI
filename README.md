# 🎾 LineSight AI: Geometric Boundary Verification Engine

LineSight AI is a lightweight, low-cost computer vision prototype designed to automate tennis boundary decisions (IN/OUT) from a single, uncalibrated 2D camera frame. 

Unlike professional multi-camera systems (e.g., Hawk-Eye) that require massive hardware to triangulate 3D space, LineSight AI utilizes deterministic geometric mathematics, robust statistical regression, and HSV color segmentation to verify physical boundaries with millimeter-level accuracy.

---

## 🚀 Core Engine Architecture

Instead of relying on unpredictable "black-box" deep learning models, LineSight AI is built as a physics-aware deterministic engine. The pipeline operates in five highly constrained phases:

### 1. Environmental Isolation (HSV Segmentation)
Standard RGB image processing fails under dynamic lighting (shadows, sun glare). LineSight AI converts the raw frame into the **HSV (Hue, Saturation, Value)** color space. By isolating the base color (Hue) from lighting intensity (Value), the engine robustly detects the yellow ball and white boundary strip regardless of court shadows.

### 2. Morphological Data Cleaning
Clay courts introduce high levels of granular noise (dirt scattered across painted lines). To prevent the engine from reading a "broken" line, we apply **Morphological Closing** matrix operations. This physically dilates and erodes the pixels, melting microscopic gaps closed to form a mathematically measurable, solid geometric contour.

### 3. Huber Loss Edge Regression (Occlusion Resistance)
Standard bounding boxes suffer from perspective distortion (the "trapezoid effect" of camera lenses). LineSight AI mathematically splits the white contour in half and extracts a 1-pixel-thick outer shell. 
* **The Problem:** When the tennis ball touches the line, it occludes the paint, causing standard Least Squares regression to skew toward the missing data.
* **The Solution:** We fit the boundary vector using **Huber Loss Optimization** (`cv2.DIST_HUBER`). This statistical model inherently identifies the ball's occlusion as a data outlier and ignores it, ensuring the boundary line remains perfectly straight.

### 4. Slant-Aware Linear Interpolation
To account for tilted camera angles, the engine calculates the exact mathematical equation of the boundary vector. It then dynamically interpolates the exact X-coordinate of the boundary line precisely at the Y-latitude of the tennis ball's impact. The system mathematically incorporates the ball's physical radius to natively respect the official tennis "Touch Rule."

### 5. Strict Dynamic Confidence Scoring
The system avoids arbitrary confidence numbers. It calculates a dynamic heuristic score (out of 100%) based on physical variables, fully visible in the **Analysis Breakdown** panel:
* **Base System Score (20%):** Awarded for successfully establishing the geometry of the court.
* **Ball Shape Integrity (40%):** Compares the active pixel area of the detected ball against a perfect circle (πr²) to heavily penalize motion blur or partial occlusion.
* **Margin Strength (40%):** Scales the confidence based on proximity to the threshold. A ball sitting on the razor's edge of the line outputs a realistically lower confidence score than a ball clearing the line by a wide margin.

---

## 💻 Premium Dashboard & UI

LineSight AI features a production-ready, dark-mode Streamlit interface utilizing custom CSS and glassmorphism styling. Key dashboard features include:
* **Final Verdict Classification:** Dynamically labels the severity of the call (e.g., `CLEAR IN`, `CLOSE CALL`, `TOUCHING`).
* **Analysis Breakdown:** Transparent progress bars showing exactly how the engine calculated the final confidence score.
* **Session Report:** Raw pixel measurements for engineers to verify the distance between the ball center and the interpolated boundary.
* **Engine Debug Mode:** A toggleable Explainable AI (XAI) view that reveals the internal morphological masks to the user.

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit app

```bash
streamlit run LineApp.py
```

---






