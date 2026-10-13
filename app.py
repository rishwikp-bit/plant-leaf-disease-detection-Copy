
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import pandas as pd
import io

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Leaf Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "plant_model_test.keras"
)

CLASS_NAMES = [
    "Apple Scab Leaf",
    "Bell_pepper leaf spot",
    "Corn leaf blight",
    "Potato leaf late blight"
]

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f8f4;
}

[data-testid="stSidebar"] {
    background: #123c2c;
}

[data-testid="stSidebar"] * {
    color: #f5fff8 !important;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 800;
    color: #ffffff;
    padding: 15px 0 4px 0;
}

.brand-sub {
    font-size: 12px;
    color: #c8dfd0;
    margin-bottom: 24px;
}

.team-tag {
    display: inline-block;
    background: #e1f3e7;
    color: #175b39;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 16px;
}

.main-title {
    font-family: 'Manrope', sans-serif;
    color: #153d2c;
    font-size: clamp(25px, 3.3vw, 38px);
    font-weight: 800;
    line-height: 1.25;
    margin: 0 0 22px 0;
}

.panel {
    background: #ffffff;
    border: 1px solid #e2eae2;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 800;
    color: #173e2d;
    margin-bottom: 7px;
}

.muted {
    color: #66786b;
    font-size: 13px;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e1e9e0;
    border-radius: 15px;
    padding: 18px;
    min-height: 105px;
}

.metric-label {
    color: #65786b;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
}

.metric-value {
    color: #17472f;
    font-size: 21px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.result-title {
    font-family: 'Manrope', sans-serif;
    color: #16472f;
    font-size: 25px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

div.stButton > button {
    background: #17603b;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #10472b;
    color: white;
    border: none;
}

hr {
    border-color: #e1e9e0;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown('<div class="brand">🌿 Plant Disease Detection</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">PLANT HEALTH CLASSIFICATION</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Project information")
    st.write("**Team:** TEAM-20")
    st.write("**Model:** MobileNetV2")
    st.write("**Framework:** TensorFlow")
    st.write("**Interface:** Streamlit")

    st.markdown("---")
    st.markdown("### How to use")
    st.write("1. Upload a leaf photograph.")
    st.write("2. Click Analyse Leaf.")
    st.write("3. Review the predicted class.")
    st.write("4. Download the report if needed.")

    st.markdown("---")
    st.caption("Predictions depend on image quality and model training.")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="team-tag">TEAM-20</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">'
    'Plant Leaf Disease Detection System using CNN Model in Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not Path(MODEL_PATH).exists():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Please check that your trained model exists at this path."
    )
    st.stop()

try:
    model = load_model()
    model_status = "Ready"
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

input_shape = model.input_shape

if isinstance(input_shape, list):
    input_shape = input_shape[0]

try:
    image_height = int(input_shape[1]) if input_shape[1] else 224
    image_width = int(input_shape[2]) if input_shape[2] else 224
except (IndexError, TypeError, ValueError):
    image_height, image_width = 224, 224

# --------------------------------------------------
# STATUS CARDS
# --------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL STATUS</div>
        <div class="metric-value">● Ready</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">CLASSIFICATION MODEL</div>
        <div class="metric-value">MobileNetV2</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">SUPPORTED CLASSES</div>
        <div class="metric-value">{len(CLASS_NAMES)}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# UPLOAD AND ANALYSIS AREA
# --------------------------------------------------

left, right = st.columns([1, 1.05], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Leaf image</div>
        <div class="muted">
            Upload a clear photograph of a single plant leaf.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a leaf photograph",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image with the leaf visible."
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded leaf photograph",
                use_container_width=True
            )

            st.caption(
                f"Image dimensions: {image.width} × {image.height} pixels"
            )
        except Exception as e:
            st.error(f"Unable to read this image: {e}")
            image = None
    else:
        image = None
        st.info("Upload an image to begin.")

with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Classification report</div>
        <div class="muted">
            The trained model predicts one of the configured classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("**Prediction settings**")
    st.write(f"Input size: {image_width} × {image_height}")
    st.write(f"Available categories: {len(CLASS_NAMES)}")

    analyse = st.button(
        "Analyse Leaf",
        disabled=image is None,
        use_container_width=True
    )

    if analyse and image is not None:
        try:
            with st.spinner("Analysing the leaf image..."):
                resized = image.resize(
                    (image_width, image_height)
                )

                arr = np.asarray(resized, dtype=np.float32)
                arr = np.expand_dims(arr, axis=0)

                # IMPORTANT:
                # This uses raw pixel values (0–255).
                # Change preprocessing only if your model was
                # trained using a different preprocessing method.

                predictions = model.predict(arr, verbose=0)
                scores = np.asarray(predictions).reshape(-1)

                if len(scores) != len(CLASS_NAMES):
                    st.error(
                        "The model output does not match the configured "
                        f"{len(CLASS_NAMES)} classes. Model outputs: "
                        f"{len(scores)}. Check your actual class names "
                        "and training class order."
                    )
                    st.stop()

                # Convert logits to probabilities when appropriate.
                if (
                    np.any(scores < 0)
                    or np.any(scores > 1)
                    or not np.isclose(np.sum(scores), 1.0, atol=0.02)
                ):
                    exp_scores = np.exp(scores - np.max(scores))
                    probabilities = exp_scores / np.sum(exp_scores)
                else:
                    probabilities = scores / np.sum(scores)

                best_index = int(np.argmax(probabilities))
                best_class = CLASS_NAMES[best_index]
                confidence = float(probabilities[best_index]) * 100

                results_df = pd.DataFrame({
                    "Plant disease category": CLASS_NAMES,
                    "Model score (%)": np.round(probabilities * 100, 2)
                }).sort_values(
                    "Model score (%)",
                    ascending=False
                )

                st.session_state["plant_result"] = {
                    "predicted_class": best_class,
                    "confidence": confidence,
                    "results": results_df,
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "filename": uploaded_file.name
                }

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# DISPLAY PREDICTION
# --------------------------------------------------

result = st.session_state.get("plant_result")

if result:
    st.write("")
    st.markdown("---")
    st.markdown("## Prediction results")

    r1, r2 = st.columns([1.3, 1])

    with r1:
        st.markdown("""
        <div class="panel">
            <div class="muted">PREDICTED CATEGORY</div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="result-title">{result["predicted_class"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(f"""
        <div class="panel">
            <div class="muted">MODEL CONFIDENCE SCORE</div>
            <div class="result-title">{result["confidence"]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    if result["confidence"] < 60:
        st.warning(
            "The model's confidence score is relatively low. "
            "Try a clearer image and treat the prediction cautiously."
        )
    else:
        st.info(
            "The score represents the model's output, not a guarantee "
            "that the classification is correct."
        )

    st.markdown("### Class-wise scores")

    chart_df = result["results"].set_index(
        "Plant disease category"
    )
    st.bar_chart(chart_df["Model score (%)"])

    st.dataframe(
        result["results"],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    report_text = f"""
PLANT LEAF DISEASE DETECTION SYSTEM USING CNN MODEL IN DEEP LEARNING
TEAM-20

Prediction report
-----------------
Image filename: {result["filename"]}
Analysis time: {result["timestamp"]}
Predicted category: {result["predicted_class"]}
Model confidence score: {result["confidence"]:.2f}%

Class-wise model scores:
{result["results"].to_string(index=False)}

Important note:
This report records the model's prediction. It is not a confirmed
agricultural diagnosis. Verify results with a qualified agricultural
expert before making treatment decisions.
"""

    st.download_button(
        label="Download prediction report",
        data=report_text,
        file_name="plant_leaf_prediction_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.caption(
        "The system classifies images using its trained model. "
        "It does not independently confirm disease severity or "
        "recommend chemical treatment."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.caption(
    "Plant Leaf Disease Detection System using CNN Model in Deep Learning · TEAM-20"
)
