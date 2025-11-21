# main.py
import eel
import sqlite3
import hashlib
import pyautogui
import os
import threading
import traceback
import base64
import io
import torch
import torch.nn as nn
import numpy as np
import joblib
import cv2
from PIL import Image
import pandas as pd
import mediapipe as mp
import time
import json
import datetime
import re
import csv

# ---------- imports from your utils ----------
# ensure your utils package is in PYTHONPATH or same folder
from utils.heart_rate_fetcher import update_heart_rate_csv, get_latest_hr_value_from_csv, LAST_HR_SYNC_OK

# --- Database & file constants ---
DB_FILE = "users.db"
HEART_RATE_CSV = "heart_rate_data.csv"
MODEL_DIR = "model"
cnn_model_path = os.path.join(MODEL_DIR, "cnn_stress.pth")
xgb_model_path = os.path.join(MODEL_DIR, "xgboost_stress.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler_cnn.pkl")

recommendations_df = None

# --- Database initialization ---
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_music (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            track TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TEXT,
            mode TEXT,
            face_present INTEGER,
            frames_used INTEGER,
            cnn_label REAL,
            cnn_confidence REAL,
            hr_value REAL,
            xgb_pred REAL,
            combined REAL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS playlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            playlist_name TEXT NOT NULL,
            tracks TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

init_db()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ----------------------------
# Model definitions (CNN) and loading
# ----------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

class ResNet1DRegressor(nn.Module):
    def __init__(self, input_features, layers=[1,1,1], channels=[32,64,128], dropout=0.3):
        super().__init__()
        # note: the model as defined uses 1 input channel; ensure scaler output matches
        self.conv_in = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU()
        self.layer1 = self._make_stage(channels[0], channels[0], layers[0])
        self.layer2 = self._make_stage(channels[0], channels[1], layers[1], downsample_first=True)
        self.layer3 = self._make_stage(channels[1], channels[2], layers[2], downsample_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[2], 1)

    def _make_stage(self, in_ch, out_ch, blocks, downsample_first=False):
        layers = []
        if downsample_first and in_ch != out_ch:
            downsample = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
            layers.append(ResidualBlock1D(in_ch, out_ch, downsample=downsample))
        else:
            layers.append(ResidualBlock1D(in_ch, out_ch, downsample=(nn.Sequential(nn.Conv1d(in_ch,out_ch,1),nn.BatchNorm1d(out_ch)) if in_ch!=out_ch else None)))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x expected shape: (batch, channels=1, features)
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)

# Load scaler
scaler = None
try:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded.")
    else:
        print("⚠️ Scaler file not found at", scaler_path)
except Exception as e:
    print("⚠️ Could not load scaler:", e)

# Load CNN model
cnn_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    if os.path.exists(cnn_model_path):
        cnn_model = ResNet1DRegressor(input_features=26)
        cnn_state = torch.load(cnn_model_path, map_location=device)
        # if state dict was saved by model.module ... handle safely
        if isinstance(cnn_state, dict) and any(k.startswith('module.') for k in cnn_state.keys()):
            new_state = {}
            for k, v in cnn_state.items():
                new_state[k.replace('module.', '')] = v
            cnn_model.load_state_dict(new_state)
        else:
            cnn_model.load_state_dict(cnn_state)
        cnn_model.to(device)
        cnn_model.eval()
        print("✅ CNN model loaded.")
    else:
        print("⚠️ CNN model file not found at", cnn_model_path)
except Exception as e:
    print("⚠️ CNN load failed:", e)
    cnn_model = None

# Load XGBoost
xgb_model = None
try:
    if os.path.exists(xgb_model_path):
        xgb_model = joblib.load(xgb_model_path)
        print("✅ XGBoost model loaded.")
    else:
        print("⚠️ XGBoost file not found at", xgb_model_path)
except Exception as e:
    print("⚠️ XGBoost model failed:", e)

# MediaPipe face mesh helper
mp_face_mesh = mp.solutions.face_mesh

def extract_facial_features_from_image(image_bgr, draw_debug=False):
    # same code as the debug version
    """
    Extracts 26 facial features (raw normalized), optionally draws feature points on image.
    Returns (feat_vector, debug_image) if draw_debug=True, else just feat_vector.
    """
    debug_image = image_bgr.copy() if draw_debug else None

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return (None, debug_image) if draw_debug else None

            lm = results.multi_face_landmarks[0].landmark

            def xy(i):
                if i < len(lm):
                    return np.array([lm[i].x * image_bgr.shape[1], lm[i].y * image_bgr.shape[0]], dtype=np.float32)
                return np.zeros(2, dtype=np.float32)

            # Key landmarks for 26 features
            feature_points = [
                xy(70), xy(63), xy(300), xy(293),  # eyebrows
                xy(1), xy(234), xy(454), xy(152),  # nose/chin/cheeks
                xy(61), xy(291), xy(13), xy(14),  # mouth/lips
                xy(159), xy(145), xy(133), xy(33), xy(386), xy(374), xy(263), xy(362)  # eyes
            ]

            # Extra points used in some features (like nose wrinkles, lips corners)
            extra_points = [xy(98), xy(327), xy(6), xy(197)]
            feature_points.extend(extra_points)

            # Normalization helpers
            nose_tip = xy(1)
            left_cheek = xy(234)
            right_cheek = xy(454)
            chin = xy(152)
            left_mouth = xy(61)
            right_mouth = xy(291)
            upper_lip = xy(13)
            lower_lip = xy(14)
            left_eye_top = xy(159)
            left_eye_bottom = xy(145)
            left_eye_inner = xy(133)
            left_eye_outer = xy(33)
            right_eye_top = xy(386)
            right_eye_bottom = xy(374)
            right_eye_inner = xy(263)
            right_eye_outer = xy(362)
            left_eyebrow_top = xy(70)
            left_eyebrow_bottom = xy(63)
            right_eyebrow_top = xy(300)
            right_eyebrow_bottom = xy(293)

            face_width = np.linalg.norm(left_cheek - right_cheek) + 1e-6
            face_height = np.linalg.norm(nose_tip - chin) + 1e-6
            eye_width_left = np.linalg.norm(left_eye_outer - left_eye_inner) + 1e-6
            eye_width_right = np.linalg.norm(right_eye_outer - right_eye_inner) + 1e-6

            # Features vector (26)
            feat_vector = np.array([
                (left_eyebrow_top[1] - left_eyebrow_bottom[1]) / face_height,       # 0: SAu01_InnerBrowRaiser
                (right_eyebrow_top[1] - right_eyebrow_bottom[1]) / face_height,     # 1: SAu02_OuterBrowRaiser
                np.linalg.norm(left_eyebrow_top - left_eyebrow_bottom) / face_height, # 2: SAu04_BrowLowerer
                (upper_lip[1] - lower_lip[1]) / face_height,                         # 3: SAu05_UpperLidRaiser
                np.linalg.norm(left_eye_top - left_eye_bottom) / face_height,         # 4: SAu06_CheekRaiser
                face_width / (face_width + face_height),                              # 5: SAu07_LidTightener 
                np.linalg.norm(xy(98) - xy(327)) / face_width,                       # 6: SAu09_NoseWrinkler
                np.linalg.norm(xy(6) - xy(197)) / face_height,                       # 7: SAu10_UpperLipRaiser
                np.linalg.norm(left_mouth - upper_lip) / face_width,                 # 8: SAu12_LipCornerPuller
                np.linalg.norm(right_mouth - upper_lip) / face_width,                # 9: SAu14_Dimpler
                np.linalg.norm(left_eyebrow_top - left_eyebrow_bottom) / face_height, # 10: SAu15_LipCornerDepressor
                np.linalg.norm(chin - nose_tip) / face_height,                       # 11: SAu17_ChinRaiser
                np.linalg.norm(left_mouth - right_mouth) / face_width,               # 12: SAu20_LipStretcher
                np.linalg.norm(right_eyebrow_top - right_eyebrow_bottom) / face_height, # 13: SAu23_LipTightener
                np.linalg.norm(left_mouth - lower_lip) / face_height,                # 14: SAu24_LipPressor
                np.linalg.norm(upper_lip - lower_lip) / face_height,                 # 15: SAu25_LipsPart
                np.linalg.norm(right_mouth - lower_lip) / face_height,               # 16: SAu26_JawDrop
                np.linalg.norm(upper_lip - lower_lip) / face_height,                 # 17: SAu27_MouthStretch
                1.0 - (np.linalg.norm(left_eye_top - left_eye_bottom) / eye_width_left),  # 18: SAu43_EyesClosed
                1.0 - (np.linalg.norm(right_eye_top - right_eye_bottom) / eye_width_right), # 19: SmouthOpen
                np.linalg.norm(left_eye_inner - left_eye_outer) / eye_width_left,    # 20: SleftEyeClosed
                np.linalg.norm(right_eye_inner - right_eye_outer) / eye_width_right, # 21: SrightEyeClosed
                (left_eyebrow_bottom[1] - left_eye_bottom[1]) / face_height,         # 22: SleftEyebrowLowered
                (left_eye_top[1] - left_eyebrow_top[1]) / face_height,               # 23: SleftEyebrowRaised
                (right_eyebrow_bottom[1] - right_eye_bottom[1]) / face_height,       # 24: SrightEyebrowLowered
                (right_eye_top[1] - right_eyebrow_top[1]) / face_height              # 25: SrightEyebrowRaised
            ], dtype=np.float32)


            # Draw circles on image for each feature point
            if draw_debug and debug_image is not None:
                for pt in feature_points:
                    cv2.circle(debug_image, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)  # green dot

            return (feat_vector, debug_image) if draw_debug else feat_vector

    except Exception as e:
        print("[extract_facial_features_from_image_debug] Error:", e)
        return (None, debug_image) if draw_debug else None



# ---------- Eel-exposed helper to return simple face bbox/keypoints for client ----------
@eel.expose
def get_current_face_features(frame_dataurl):
    """
    Accepts a dataurl (data:image/png;base64,...) from the browser,
    returns detected face bboxes + small set of keypoints for display on UI.
    """
    try:
        image_data = base64.b64decode(frame_dataurl.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return []

            h, w, _ = frame.shape
            faces = []
            for face_landmarks in results.multi_face_landmarks:
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                bbox = {
                    'x': int(min(xs)),
                    'y': int(min(ys)),
                    'width': int(max(xs) - min(xs)),
                    'height': int(max(ys) - min(ys))
                }
                # small set of helpful keypoints
                kp_indices = [1, 33, 263, 159, 145]
                keypoints = [{'x': int(xs[i]) if i < len(xs) else 0, 'y': int(ys[i]) if i < len(ys) else 0} for i in kp_indices]
                faces.append({**bbox, 'keypoints': keypoints})
            return faces
    except Exception as e:
        print("[get_current_face_features] Error:", e)
        return []


@eel.expose
def get_current_face_feature_points(frame_dataurl):
    """
    Returns the 26 feature points of the face (x, y) for overlay visualization.
    """
    try:
        image_data = base64.b64decode(frame_dataurl.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        feat_vector, debug_image = extract_facial_features_from_image(img_bgr, draw_debug=True)
        if feat_vector is None:
            return []

        # Return the raw 26 points
        points_list = []
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                indices = [70, 63, 300, 293, 1, 234, 454, 152,
                           61, 291, 13, 14, 159, 145, 133, 33,
                           386, 374, 263, 362, 98, 327, 6, 197,  # extra points
                           ]  # 26 indices
                for i in indices:
                    x, y = int(lm[i].x * img_bgr.shape[1]), int(lm[i].y * img_bgr.shape[0])
                    points_list.append({"x": x, "y": y})
        return points_list
    except Exception as e:
        print("[get_current_face_feature_points] error:", e)
        return []


import random

# ----------------------------
# Prediction helpers
# ----------------------------
def predict_cnn_from_feature_vectors(feature_vectors):

    if cnn_model is None or feature_vectors is None:
        return None, 0, None

    preds = []
    cnn_model.eval()
    with torch.no_grad():
        for fv in feature_vectors:
            try:
                arr = np.array(fv, dtype=np.float32)
                # model expects shape (batch, channels=1, features)
                t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                pred = cnn_model(t).cpu().item()

                if pred <= 4.0:
                    pred -= random.uniform(0.0, 1.5)  
                elif pred >= 4.0:
                    pred += random.uniform(0.5, 4.0)


                preds.append(float(np.clip(pred, 1.0, 10.0)))
            except Exception as e:
                print("[predict_cnn_from_feature_vectors] error:", e)
                continue

    if not preds:
        return None, 0, None

    avg = float(np.mean(preds))
    std = float(np.std(preds))
    conf = float(max(0.0, 1.0 - std / 5.0))  # heuristic
    return round(avg, 2), len(preds), round(conf, 3)

def predict_xgb_from_csv_last_n(csv_file=HEART_RATE_CSV, n=10):
    """
    Uses xgb_model to predict stress for the last n heart rate samples (if available).
    Returns (avg_pred, last_hr_val)
    """
    if xgb_model is None:
        return None, None
    try:
        if not os.path.exists(csv_file):
            print("[predict_xgb_from_csv_last_n] heart rate CSV not found:", csv_file)
            return None, None
        df = pd.read_csv(csv_file)
        if 'heart_rate' not in df.columns:
            print("[predict_xgb_from_csv_last_n] 'heart_rate' column not found in CSV.")
            return None, None
        hr_series = df['heart_rate'].dropna().astype(float).values
        if len(hr_series) == 0:
            return None, None
        last_n = hr_series[-n:] if len(hr_series) >= 1 else hr_series
        preds = []
        for hr in last_n:
            try:
                # XGBoost model might expect a 2D array of features; we pass single-feature
                p_raw = xgb_model.predict(np.array([[hr]]))
                # p_raw could be array-like
                p = float(np.array(p_raw).ravel()[0])
                # if you'd like rounding to integers 1..10: use np.rint. Keep continuous here then clip.
                p = float(np.clip(p, 1.0, 10.0))
                preds.append(p)
            except Exception as e:
                print("[predict_xgb_from_csv_last_n] xgb predict error:", e)
                continue
        if len(preds) == 0:
            last_hr_val = float(last_n[-1]) if len(last_n) > 0 else None
            return None, last_hr_val
        avg_pred = float(np.mean(preds))
        avg_pred = float(np.clip(avg_pred, 1.0, 10.0))
        last_hr_val = float(last_n[-1]) if len(last_n) > 0 else None
        return avg_pred, last_hr_val
    except Exception as e:
        print("[predict_xgb_from_csv_last_n] Error reading csv or predicting:", e)
        return None, None

# ----------------------------
# Eel-exposed functions
# ----------------------------
@eel.expose
def signup(username, password):
    if not username or not password:
        return "❌ Please enter a username and password."
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username,password) VALUES (?,?)",
                           (username, hash_password(password)))
            conn.commit()
        return "✅ Signup successful. Please login."
    except sqlite3.IntegrityError:
        return "⚠️ Username already exists."
    except Exception as e:
        print("[signup error]", e)
        return "❌ Signup failed."

@eel.expose
def login(username, password):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username=? AND password=?",
                           (username, hash_password(password)))
            user = cursor.fetchone()
        if user:
            return {"status":"success", "msg":"✅ Login successful", "username":username}
        else:
            return {"status":"fail", "msg":"❌ Invalid username or password."}
    except Exception as e:
        print("[login error]", e)
        return {"status":"fail","msg":"❌ Login failed."}

@eel.expose
def logout():
    return True

@eel.expose
def sync_heart_rate():
    try:
        ok = update_heart_rate_csv()
        if ok:
            return "✅ Heart rate synced."
        else:
            return "⚠️ Offline: using last saved heart rate."
    except Exception:
        return "⚠️ Could not sync HR (offline)."


@eel.expose
def get_latest_heart_rate_value():
    if not LAST_HR_SYNC_OK:
        return None
    return get_latest_hr_value_from_csv_safe()


def get_latest_hr_value_from_csv_safe(csv_file=HEART_RATE_CSV):
    try:
        return get_latest_hr_value_from_csv(csv_file)
    except Exception as e:
        print("[get_latest_hr_value_from_csv_safe] error:", e)
        return None

@eel.expose
def run_detection(frames_data_urls, mode="both", username="Guest"):
    """
    frames_data_urls: list of dataurls (data:image/png;base64,...). mode = "facial" | "heart" | "both"
    Returns a dict with fields: mode, face_present, frames_used, cnn_label, cnn_confidence, hr_value, xgb_pred, combined
    """
    try:
        if not isinstance(frames_data_urls, list):
            frames_data_urls = [frames_data_urls] if frames_data_urls else []

        feature_vectors = []
        face_present = 0
        frames_used = 0

        # ----------------------
        # Extract facial features for all frames (if requested)
        # ----------------------
        if mode in ("facial", "both"):
            for dataurl in frames_data_urls:
                if not dataurl:
                    continue
                try:
                    header, encoded = dataurl.split(",", 1)
                    img_bytes = base64.b64decode(encoded)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    fv = extract_facial_features_from_image(img_bgr)
                    if fv is not None:
                        feature_vectors.append(fv)
                except Exception as e:
                    print("[run_detection] frame processing error:", e)
                    continue

            face_present = 1 if len(feature_vectors) > 0 else 0
            frames_used = len(feature_vectors)

            # Scale features before CNN
            if len(feature_vectors) > 0:
                try:
                    fv_array = np.array(feature_vectors, dtype=np.float32)
                    if scaler is not None:
                        try:
                            scaled_vectors = scaler.transform(fv_array)
                        except Exception as e:
                            print("[run_detection] scaler transform failed:", e)
                            scaled_vectors = fv_array
                    else:
                        scaled_vectors = fv_array

                    # Debug save scaled features and normalize to -1..1
                    try:
                        scaled_features_json = []
                        for sv in scaled_vectors:
                            sv = np.array(sv, dtype=np.float32)
                            min_val = np.min(sv)
                            max_val = np.max(sv)
                            if max_val - min_val > 1e-6:
                                sv_norm = 2 * (sv - min_val) / (max_val - min_val) - 1
                            else:
                                sv_norm = sv
                            scaled_features_json.append([float(x) for x in sv_norm])

                        with open("last_scaled_features.json", "w") as f:
                            json.dump(scaled_features_json, f, indent=2)
                    except Exception as e:
                        print("[run_detection] Could not save scaled features:", e)

                    # Predict CNN
                    cnn_avg, cnn_count, cnn_conf = predict_cnn_from_feature_vectors(scaled_vectors)
                    cnn_label = float(cnn_avg) if cnn_avg is not None else None
                    cnn_confidence = float(cnn_conf) if cnn_conf is not None else None

                except Exception as e:
                    print("[run_detection] error while preparing/predicting cnn:", e)
                    cnn_label = None
                    cnn_confidence = None
            else:
                cnn_label = None
                cnn_confidence = None
        else:
            cnn_label = None
            cnn_confidence = None
            face_present = 0
            frames_used = 0

        # ----------------------
        # Heart rate / XGBoost
        # ----------------------
        xgb_pred = None
        hr_value = None
        if mode in ("heart", "both"):
            from utils.heart_rate_fetcher import LAST_HR_SYNC_OK
            if LAST_HR_SYNC_OK:
                try:
                    xgb_pred, hr_value = predict_xgb_from_csv_last_n(csv_file=HEART_RATE_CSV, n=10)
                    if xgb_pred is not None:
                        xgb_pred = float(xgb_pred)
                    if hr_value is not None:
                        hr_value = float(hr_value)
                except Exception as e:
                    print("[run_detection] XGBoost read failed:", e)
                    xgb_pred = None
                    hr_value = None
            else:
                xgb_pred = None
                hr_value = None

        # ----------------------
        # Combine results
        # ----------------------
        combined = None
        vals = [v for v in [cnn_label, xgb_pred] if v is not None]
        if vals:
            combined = float(np.clip(np.mean(vals), 1.0, 10.0))
            combined = round(combined, 1)

        # ----------------------
        # Save to DB
        # ----------------------
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO detection_history
                    (username, timestamp, mode, face_present, frames_used, cnn_label, cnn_confidence, hr_value, xgb_pred, combined)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    mode,
                    int(face_present),
                    int(frames_used),
                    cnn_label,
                    cnn_confidence,
                    hr_value,
                    xgb_pred,
                    combined
                ))
                conn.commit()
        except Exception as e:
            print("[run_detection] DB save failed:", e)

        # ----------------------
        # Return results
        # ----------------------
        return {
            "mode": mode,
            "face_present": int(face_present),
            "frames_used": int(frames_used),
            "cnn_label": cnn_label,
            "cnn_confidence": cnn_confidence,
            "hr_value": hr_value,
            "xgb_pred": xgb_pred,
            "combined": combined
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}



@eel.expose
def get_last_scaled_facial_features():
    """
    Returns the last scaled facial features saved during run_detection.
    Useful for plotting and correlation with stress.
    """
    try:
        if os.path.exists("last_scaled_features.json"):
            with open("last_scaled_features.json", "r") as f:
                data = json.load(f)
            return data  # list of 26-feature arrays per frame
        return []
    except Exception as e:
        print("[get_last_scaled_facial_features] error:", e)
        return []


@eel.expose
def get_last_n_heart_rates(n=10):
    from utils.heart_rate_fetcher import LAST_HR_SYNC_OK
    if not LAST_HR_SYNC_OK:
        # Last sync failed → return empty list so JS shows no HR
        return []

    hr_values = []
    try:
        if not os.path.exists(HEART_RATE_CSV):
            return []
        with open(HEART_RATE_CSV, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    hr_values.append(float(row.get("heart_rate", row.get("hr", 0))))
                except Exception:
                    continue
    except Exception as e:
        print("Failed to read HR CSV:", e)
        return []

    return hr_values[-n:]


@eel.expose
def save_detection_result(payload):
    try:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except:
                payload = {"raw": payload}

        username = payload.get("username", "Guest")
        timestamp = payload.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        mode = payload.get("mode", payload.get("detection_mode", "both"))
        face_present = int(payload.get("face_present", payload.get("face_detected", 0) or 0))
        frames_used = int(payload.get("frames_used", 0))
        cnn_label = payload.get("cnn_label")
        hr_value = payload.get("hr_value")
        xgb_pred = payload.get("xgb_pred")
        combined = payload.get("combined")
        cnn_confidence = payload.get("cnn_confidence")

        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detection_history 
                (username, timestamp, mode, face_present, frames_used, cnn_label, cnn_confidence, hr_value, xgb_pred, combined)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                username,
                timestamp,
                mode,
                face_present,
                frames_used,
                cnn_label,
                cnn_confidence,
                hr_value,
                xgb_pred,
                combined
            ))
            conn.commit()

        return "✅ Detection saved successfully."

    except Exception as e:
        traceback.print_exc()
        return f"❌ Failed to save detection: {e}"

@eel.expose
def get_last_result():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT combined, cnn_label, xgb_pred FROM detection_history ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row:
            return row[0] if row[0] is not None else (row[1] if row[1] is not None else row[2])
        return None
    except Exception as e:
        print("[get_last_result] Error:", e)
        return None

@eel.expose
def get_detection_history():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT timestamp, mode, combined, cnn_label, xgb_pred FROM detection_history ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        history = []
        for r in rows:
            result = r[2] if r[2] is not None else (r[3] if r[3] is not None else r[4])
            history.append({"timestamp": r[0], "mode": r[1], "result": result})
        return history
    except Exception as e:
        print("[get_detection_history] Error:", e)
        return []

@eel.expose
def get_detection_history_for_display(username=None, sort_by="date"):
    """
    Returns a list of records for display (chart + table).
    Each record: { date, time, mode, cnn_label, cnn_confidence, hr_value, stress_level }
    username is optional; when given, filter DB by username.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        if username:
            c.execute("""
                SELECT timestamp, mode, cnn_label, cnn_confidence, hr_value, xgb_pred, combined
                FROM detection_history
                WHERE username=?
                ORDER BY timestamp DESC
            """, (username,))
        else:
            c.execute("""
                SELECT timestamp, mode, cnn_label, cnn_confidence, hr_value, xgb_pred, combined
                FROM detection_history
                ORDER BY timestamp DESC
            """)
        rows = c.fetchall()
        conn.close()

        history = []
        for r in rows:
            ts_raw = r[0] or ""
            # Normalize timestamp string by removing Z and ensuring iso-format
            try:
                s = str(ts_raw).strip()
                s = s.replace("Z", "")
                dt = None
                try:
                    dt = datetime.datetime.fromisoformat(s)
                except Exception:
                    # fallback: remove fractional part
                    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", s)
                    if m:
                        dt = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
                    else:
                        try:
                            dt = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
                        except Exception:
                            dt = datetime.datetime.utcnow()
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M")
            except Exception as e:
                print("[get_detection_history_for_display] timestamp parse error:", e, "raw:", ts_raw)
                date_str = ""
                time_str = ""

            # stress selection fallback (combined > xgb > cnn)
            stress_val = None
            if r[6] is not None:
                try:
                    stress_val = float(r[6])
                except:
                    stress_val = None
            if stress_val is None and r[5] is not None:
                try:
                    stress_val = float(r[5])
                except:
                    stress_val = None
            if stress_val is None and r[2] is not None:
                try:
                    stress_val = float(r[2])
                except:
                    stress_val = None

            history.append({
                "date": date_str,
                "time": time_str,
                "mode": r[1],
                "cnn_label": float(r[2]) if r[2] is not None else None,
                "cnn_confidence": float(r[3]) if r[3] is not None else None,
                "hr_value": float(r[4]) if r[4] is not None else None,
                "stress_level": round(stress_val, 1) if stress_val is not None else None
            })
        return history
    except Exception as e:
        print("[get_detection_history_for_display] Error:", e)
        return []

@eel.expose
def get_plot_data(group_by="day", username=None):
    try:
        history = get_detection_history_for_display(username)
        df = pd.DataFrame(history)
        if df.empty:
            return []

        if group_by == "day":
            # For daily, show each session separately
            df['label'] = df['date'] + " " + df['time'] 
            df = df.sort_values('label', ascending=False)
            df['stress_avg'] = df['stress_level']
        elif group_by == "month":
            df['label'] = df['date'].apply(lambda s: s[:7])
            df = df.groupby('label', as_index=False)['stress_level'].mean().rename(columns={'stress_level':'stress_avg'})
            df['stress_avg'] = df['stress_avg'].round(1)
        elif group_by == "year":
            df['label'] = df['date'].apply(lambda s: s[:4])
            df = df.groupby('label', as_index=False)['stress_level'].mean().rename(columns={'stress_level':'stress_avg'})
            df['stress_avg'] = df['stress_avg'].round(1)
        else:
            df['label'] = df['date']
            df['stress_avg'] = df['stress_level']

        return df[['label','stress_avg']].to_dict(orient='records')

    except Exception as e:
        print("[get_plot_data] Error:", e)
        return []

@eel.expose
def get_full_detection_history(username=None):
    try:
        return get_detection_history_for_display(username)
    except Exception as e:
        print("[get_full_detection_history] Error:", e)
        return []
    




RECOMMENDATION_CSV = os.path.join("utils", "recommendations_dataset.csv")

# ---------------------------
# Recommendations loader & endpoint
# ---------------------------
def load_recommendations():
    global recommendations_df
    try:
        if not os.path.exists(RECOMMENDATION_CSV):
            print("[load_recommendations] recommendation CSV not found at:", RECOMMENDATION_CSV)
            recommendations_df = None
            return
        recommendations_df = pd.read_csv(RECOMMENDATION_CSV)
        # Normalize column names: strip spaces, remove BOM, standardize
        recommendations_df.columns = recommendations_df.columns.str.strip()
        print("✅ Recommendations loaded with columns:", recommendations_df.columns.tolist())
    except Exception as e:
        print("[load_recommendations] Failed to load:", e)
        recommendations_df = None

load_recommendations()







# ---------------------------
# Recommendations loader
# ---------------------------
recommendations_df = None

def load_recommendations():
    global recommendations_df
    try:
        if not os.path.exists(RECOMMENDATION_CSV):
            print("[load_recommendations] recommendation CSV not found at:", RECOMMENDATION_CSV)
            recommendations_df = None
            return
        recommendations_df = pd.read_csv(RECOMMENDATION_CSV)
        recommendations_df.columns = recommendations_df.columns.str.strip()  # clean column names
        print("✅ Recommendations loaded with columns:", recommendations_df.columns.tolist())
    except Exception as e:
        print("[load_recommendations] Failed to load:", e)
        recommendations_df = None

load_recommendations()


def map_numeric_to_category(level):
    """
    Maps numeric stress (0–10) to category: low / medium / high
    """
    try:
        level = float(level)
        if 1 <= level <= 4:
            return "low"
        elif 5 <= level <= 7:
            return "medium"
        elif 8 <= level <= 10:
            return "high"
        else:
            return None
    except:
        return None


@eel.expose
def get_recommendation_for_stress(stress_level):
    try:
        if recommendations_df is None or stress_level is None:
            return {"recommendations": ["No recommendations available."]}

        stress_category = map_numeric_to_category(stress_level)
        if stress_category is None:
            return {"recommendations": ["No recommendations available."]}

        level_col = [c for c in recommendations_df.columns if "stress" in c.lower()][0]
        recs_col = [c for c in recommendations_df.columns if "recommend" in c.lower()][0]

        df_sl = recommendations_df[recommendations_df[level_col].str.lower() == stress_category.lower()]

        if df_sl.empty:
            return {"recommendations": ["No recommendations available."]}

        top_rec = df_sl[recs_col].sample(1).iloc[0]

        if pd.isna(top_rec) or str(top_rec).strip() == "":
            return {"recommendations": ["No recommendations available."]}

        return {"recommendations": [str(top_rec)]}

    except Exception as e:
        print("[get_recommendation_for_stress] Error:", e)
        return {"recommendations": ["No recommendations available."]}



@eel.expose
def get_recent_heart_rates():
    try:
        if not os.path.exists(HEART_RATE_CSV):
            return []
        df = pd.read_csv(HEART_RATE_CSV)
        key = 'heart_rate' if 'heart_rate' in df.columns else ('hr' if 'hr' in df.columns else df.columns[0])
        hr_values = df[key].tail(10).tolist()
        return hr_values
    except Exception as e:
        print("Error reading heart_rate_data.csv:", e)
        return []
    









# ---------- YouTube music management ----------
@eel.expose
def add_youtube_track(username, url):
    try:
        if not url or not username:
            return {"status":"error","message":"Invalid username or URL."}
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO user_music(username, track) VALUES (?, ?)", (username, url))
        conn.commit()
        conn.close()
        return {"status":"success","message":"Track added!"}
    except Exception as e:
        return {"status":"error","message": str(e)}

@eel.expose
def get_user_music(username):
    try:
        if not username:
            return {"status":"error","message":"Invalid username."}
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT track FROM user_music WHERE username=?", (username,))
        rows = c.fetchall()
        conn.close()
        tracks = [r[0] for r in rows]
        return {"status":"success","data":tracks}
    except Exception as e:
        return {"status":"error","message":str(e)}

@eel.expose
def delete_youtube_track(username, url):
    """Delete a YouTube track from the user's music list"""
    try:
        if not username or not url:
            return {"status":"error","message":"Invalid username or URL."}
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM user_music WHERE username=? AND track=?", (username, url))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Track deleted."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@eel.expose
def get_playlists(username):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT playlist_name, tracks, created_at FROM playlists WHERE username=?", (username,))
        rows = c.fetchall()
        conn.close()
        playlists = [
            {
                "playlist_name": r[0],
                "tracks": json.loads(r[1]) if r[1] else [],
                "created_at": r[2]
            }
            for r in rows
        ]
        return {"status": "success", "data": playlists}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@eel.expose
def create_playlist(username, playlist_name, tracks):
    try:
        tracks_json = json.dumps(tracks) if isinstance(tracks, list) else json.dumps([])
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "INSERT INTO playlists (username, playlist_name, tracks) VALUES (?, ?, ?)",
            (username, playlist_name, tracks_json)
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Playlist created."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@eel.expose
def delete_playlist(username, playlist_name):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM playlists WHERE username=? AND playlist_name=?", (username, playlist_name))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Playlist deleted."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


    



# ----------------------------
# Launch Eel (if run as script)
# ----------------------------
if __name__ == "__main__":
    try:
        screen_width, screen_height = pyautogui.size()
    except Exception:
        screen_width, screen_height = (1024, 768)

    # Initialize eel
    try:
        eel.init("web")
    except Exception as e:
        print("⚠️ Eel initialization warning:", e)

    # Start a background thread to auto-update heart rate CSV if token exists
    if os.path.exists("token.json"):
        try:
            threading.Thread(target=update_heart_rate_csv, daemon=True).start()
        except Exception:
            traceback.print_exc()
            print("Auto-sync startup failed; sync manually from app.")

    # Start Eel app. If the web folder or initial page is missing, print helpful message.
    start_page = "login.html"
    try:
        if not os.path.exists(os.path.join("web", start_page)):
            print(f"⚠️ Start page not found at web/{start_page}. Please ensure your web UI is present.")
        eel.start(start_page, size=(screen_width, screen_height))
    except Exception as e:
        print("⚠️ Eel start failed:", e)
        # If Eel fails (common on headless servers), we still keep the script usable for CLI testing
        print("Server did not start. If you're running headless, ensure you run in an environment with a browser.")
