import streamlit as st
import cv2
import numpy as np
import math

# ---------------- CONFIG ----------------
st.set_page_config(page_title="LineSight AI | Demo", layout="wide")
st.title("🎾 LineSight AI: Line Verification Demo")

side_mode = st.sidebar.selectbox(
    "Calibration Mode",
    ["Right Side of Court", "Left Side of Court"]
)

show_debug = st.sidebar.checkbox("Show Debug View")

uploaded_file = st.file_uploader("Upload Court Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # ---------------- LOAD IMAGE ----------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    h, w, _ = img.shape
    scale = 1000 / w
    img = cv2.resize(img, (1000, int(h * scale)))
    h, w, _ = img.shape
    display_img = img.copy()

    col1, col2 = st.columns([2,1])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # =====================================================
    # 1. BALL DETECTION
    # =====================================================

    lower_yellow = np.array([10, 80, 80])
    upper_yellow = np.array([50, 255, 255])

    ball_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    ball_mask = cv2.medianBlur(ball_mask, 7)

    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_balls = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)

        if 6 < r < 200:
            valid_balls.append((int(x), int(y), int(r), area))

    if not valid_balls:
        with col1:
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        with col2:
            st.warning("Ball not detected")
        st.stop()

    best_ball = max(valid_balls, key=lambda b: b[3])
    cx, cy, ball_radius, actual_area = best_ball
    ball_center = (cx, cy)

    # =====================================================
    # 2. WHITE STRIP DETECTION
    # =====================================================

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        np.ones((5,5), np.uint8)
    )

    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    best_contour = None
    best_length = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000:
            continue

        rect = cv2.minAreaRect(c)
        (rcx, rcy), (w_rect, h_rect), angle = rect

        length = max(w_rect, h_rect)
        width = min(w_rect, h_rect)

        if length > 200 and width < 80:
            if length > best_length:
                best_length = length
                best_contour = c

    if best_contour is None:
        with col1:
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        with col2:
            st.warning("White strip not detected")
        st.stop()

    # =====================================================
    # 3. ROW-BY-ROW EDGE EXTRACTION
    # =====================================================

    pts = best_contour.reshape(-1, 2)

    y_dict = {}
    for p in pts:
        px, py = p[0], p[1]
        if py not in y_dict:
            y_dict[py] = []
        y_dict[py].append(px)

    edge_pts_list = []
    for py, x_list in y_dict.items():
        if side_mode == "Right Side of Court":
            edge_pts_list.append([max(x_list), py])
        else:
            edge_pts_list.append([min(x_list), py])

    edge_pts = np.array(edge_pts_list, dtype=np.int32)

    [vx, vy, x0, y0] = cv2.fitLine(edge_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy = float(vx[0]), float(vy[0])
    x0, y0 = float(x0[0]), float(y0[0])

    if vy != 0:
        t1 = (0 - y0) / vy
        x1 = int(x0 + t1 * vx)
        y1 = 0
        t2 = (h - y0) / vy
        x2 = int(x0 + t2 * vx)
        y2 = h
    else:
        x1, y1 = 0, int(y0)
        x2, y2 = w, int(y0)

    cv2.line(display_img, (x1,y1), (x2,y2), (0,255,0), 4)
    cv2.circle(display_img, ball_center, ball_radius, (255,0,0), 3)

    # =====================================================
    # 4. DECISION LOGIC
    # =====================================================

    if y1 != y2:
        boundary_x_at_ball = x1 + (x2 - x1) * ((cy - y1) / (y2 - y1))
    else:
        boundary_x_at_ball = x1

    cv2.line(display_img, (cx, cy), (int(boundary_x_at_ball), cy), (0,255,255), 2)

    if side_mode == "Right Side of Court":
        is_in = cx <= boundary_x_at_ball + ball_radius
    else:
        is_in = cx >= boundary_x_at_ball - ball_radius

    # =====================================================
    # 5. DISTANCE CALCULATION
    # =====================================================

    raw_offset = abs(cx - boundary_x_at_ball) - ball_radius
    distance_from_line = max(0, raw_offset)

    # =====================================================
    # 6. DECISION CLASSIFICATION (CLEAN LOGIC)
    # =====================================================

    if is_in:
        if distance_from_line <= 3:
            decision_label = "TOUCHING"
        elif distance_from_line <= 15:
            decision_label = "CLOSE CALL"
        else:
            decision_label = "CLEAR IN"
    else:
        if distance_from_line <= 3:
            decision_label = "TOUCHING (OUT)"
        elif distance_from_line <= 15:
            decision_label = "CLOSE OUT"
        else:
            decision_label = "CLEAR OUT"

    # =====================================================
    # 7. CONFIDENCE BREAKDOWN
    # =====================================================

    base_score = 20.0

    perfect_circle_area = math.pi * (ball_radius ** 2)
    circularity_ratio = min(1.0, actual_area / perfect_circle_area) if perfect_circle_area > 0 else 0
    shape_score = (circularity_ratio ** 2) * 40.0

    safe_margin_pixels = 15.0
    margin_ratio = min(1.0, distance_from_line / safe_margin_pixels)
    margin_score = margin_ratio * 40.0

    computed_confidence = base_score + shape_score + margin_score

    # =====================================================
    # 8. DISPLAY
    # =====================================================

    with col1:
        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                 caption="Final Verification Output")

    with col2:

        if is_in:
            st.success("DECISION: IN")
        else:
            st.error("DECISION: OUT")

        st.info(f"Decision Type: {decision_label}")

        st.markdown("---")

        st.metric("OVERALL CONFIDENCE", f"{computed_confidence:.1f}%")

        st.markdown("### Confidence Breakdown")

        st.write(f"Base System Score: {base_score:.1f} / 20")
        st.progress(base_score / 20)

        st.write(f"Ball Shape Integrity: {shape_score:.1f} / 40")
        st.progress(shape_score / 40)

        st.write(f"Decision Margin Strength: {margin_score:.1f} / 40")
        st.progress(margin_score / 40)

        st.markdown("---")

        st.write(f"Ball Center X: {cx} px")
        st.write(f"Boundary X: {boundary_x_at_ball:.2f} px")
        st.write(f"Distance From Line: {distance_from_line:.2f} px")

    # =====================================================
    # 9. DEBUG VIEW
    # =====================================================

    if show_debug:
        st.divider()
        d1, d2 = st.columns(2)

        with d1:
            st.image(white_mask, caption="White Mask")

        with d2:
            st.image(ball_mask, caption="Ball Mask")