import streamlit as st
import cv2
import numpy as np
import math

# --- 1. PREMIUM UI CONFIG & TENNIS STYLING ---
st.set_page_config(page_title="LineSight AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp {
        background-color: #0B0E11;
        background-image: 
            linear-gradient(rgba(0, 255, 102, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 102, 0.03) 1px, transparent 1px);
        background-size: 60px 60px;
    }

    /* REMOVE EMPTY ARTIFACTS */
    div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }
    
    /* MASSIVE LOGO EXPANSION */
    .logo-container { text-align: center; padding: 100px 0 60px 0; }
    .logo-main {
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        font-size: 8rem !important; 
        letter-spacing: -6px;
        margin: 0;
        line-height: 0.8;
        color: #FFFFFF;
    }
    .green-glow {
        color: #00FF66;
        text-shadow: 0 0 50px rgba(0,255,102,0.6);
    }

    /* GLASSMORPHISM BOX FIX */
    div[data-testid="stElementContainer"] > div[style*="border: 1px solid"] {
        background: rgba(22, 27, 34, 0.8) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(48, 54, 61, 0.8) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin-bottom: 10px !important;
    }

    /* METRIC & PROGRESS BAR STYLING */
    [data-testid="stMetricValue"] { color: #00FF66 !important; font-size: 3.5rem !important; font-weight: 900 !important; }
    .stProgress > div > div > div > div { background-color: #00FF66 !important; }
    p, span, label { color: #E2E8F0 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE HERO SECTION ---
st.markdown("""
    <div class="logo-container">
        <h1 class="logo-main"><span class="green-glow">LINE</span>SIGHT</h1>
        <p style="color: #8B949E; font-size: 1.4rem; letter-spacing: 5px; text-transform: uppercase; margin-top: 20px;">
            Precision Optic Verification ● Tennis AI
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 3. WORKSPACE LAYOUT ---
col_left, col_right = st.columns([2.2, 1], gap="large")

with col_right:
    with st.container(border=True):
        st.markdown("**SYSTEM CALIBRATION**")
        side_mode = st.radio("Target Side", ["Right Side of Court", "Left Side of Court"], horizontal=True, label_visibility="collapsed")
        show_debug = st.toggle("Engine Debug Mode", value=False)

with col_left:
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if not uploaded_file:
        st.markdown('<div style="height: 400px; display: flex; align-items: center; justify-content: center; border: 2px dashed #30363D; border-radius: 24px; color: #8B949E; background: rgba(22, 27, 34, 0.3);">READY FOR COURT CAPTURE</div>', unsafe_allow_html=True)

# --- 4. ENGINE LOGIC ---
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    scale = 1000 / w
    img = cv2.resize(img, (1000, int(h * scale)))
    h_new, w_new, _ = img.shape
    display_img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Ball Detection
    ball_mask = cv2.medianBlur(cv2.inRange(hsv, np.array([10, 80, 80]), np.array([50, 255, 255])), 7)
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_balls = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= 80:
            (bx, by), br = cv2.minEnclosingCircle(c)
            if 6 < br < 200: valid_balls.append((int(bx), int(by), int(br), area))

    if valid_balls:
        cx, cy, br, actual_area = max(valid_balls, key=lambda b: b[3])
        
        # Line Detection
        white_mask = cv2.morphologyEx(cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255])), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        best_contour, best_length = None, 0
        for c in cnts:
            if cv2.contourArea(c) < 1000: continue
            length = max(cv2.minAreaRect(c)[1])
            if length > 200 and length > best_length: best_length, best_contour = length, c

        if best_contour is not None:
            # Calculation & fitLine
            pts = best_contour.reshape(-1, 2)
            y_dict = {}
            for px, py in pts: y_dict.setdefault(py, []).append(px)
            edge_pts = np.array([[max(xl) if side_mode == "Right Side of Court" else min(xl), py] for py, xl in y_dict.items()], dtype=np.int32)
            [vx, vy, x0, y0] = cv2.fitLine(edge_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
            t1, t2 = (0 - y0[0]) / vy[0], (h_new - y0[0]) / vy[0]
            p1, p2 = (int(x0[0] + t1 * vx[0]), 0), (int(x0[0] + t2 * vx[0]), h_new)
            bx_ball = p1[0] + (p2[0] - p1[0]) * ((cy - p1[1]) / (p2[1] - p1[1])) if p1[1] != p2[1] else p1[0]

            # Logic & Distance
            is_in = cx <= bx_ball + br if side_mode == "Right Side of Court" else cx >= bx_ball - br
            dist = max(0, abs(cx - bx_ball) - br)

            # Decision Classification
            if is_in:
                decision_label = "TOUCHING" if dist <= 3 else "CLOSE CALL" if dist <= 15 else "CLEAR IN"
            else:
                decision_label = "TOUCHING (OUT)" if dist <= 3 else "CLOSE OUT" if dist <= 15 else "CLEAR OUT"

            # Breakdown Score Math
            base_score = 20.0
            shape_score = (min(1.0, actual_area / (math.pi * br**2)) ** 2) * 40.0
            margin_score = min(1.0, dist / 15.0) * 40.0
            conf = base_score + shape_score + margin_score

            # Visualization
            cv2.line(display_img, p1, p2, (0, 255, 102), 5)
            cv2.circle(display_img, (cx, cy), br, (204, 255, 0), 4)

            # --- 5. UI DISPLAY ---
            with col_left:
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col_right:
                # VERDICT BOX
                with st.container(border=True):
                    st.markdown("**FINAL VERDICT**")
                    if is_in: st.success(f"🟢 {decision_label}") #
                    else: st.error(f"🔴 {decision_label}") #
                
                # CONFIDENCE BOX
                with st.container(border=True):
                    st.markdown("**CONFIDENCE SCORE**")
                    st.metric("", f"{conf:.1f}%")
                    st.progress(conf / 100)
                
                # ANALYSIS BREAKDOWN BOX
                with st.container(border=True):
                    st.markdown("**ANALYSIS BREAKDOWN**")
                    st.caption(f"Base System Score: {base_score:.1f} / 20")
                    st.progress(base_score / 20)
                    st.caption(f"Ball Shape Integrity: {shape_score:.1f} / 40")
                    st.progress(shape_score / 40)
                    st.caption(f"Margin Strength: {margin_score:.1f} / 40")
                    st.progress(margin_score / 40)

                # NEW: PIXEL REPORT BOX
                with st.container(border=True):
                    st.markdown("**SESSION REPORT**")
                    st.write(f"Ball Center: {cx}px")
                    st.write(f"Line Offset: {dist:.2f}px")

                if show_debug:
                    st.image(white_mask, caption="DEBUG: LINE MASK")