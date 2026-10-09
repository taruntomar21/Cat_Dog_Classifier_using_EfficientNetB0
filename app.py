import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


MODEL_PATH = Path(__file__).parent / "best_efficientnet.keras"
AUTHOR = "Tarun"

# Put your own measured accuracy here (e.g. 0.97 for 97%) from your Colab
# evaluation of best_efficientnet.keras on the test set. When set, it appears in
# the "About Project" results table. Leave as None to hide it.
CUSTOM_ACCURACY = 0.9916

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .hero {
            text-align: center;
            padding: 1.4rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #a1c4fd 100%);
            margin-bottom: 1.2rem;
        }
        .hero h1 { color: #1f2937; margin: 0; font-size: 2.2rem; }
        .hero p  { color: #374151; margin: 0.4rem 0 0 0; font-size: 1.05rem; }

        .result-card {
            text-align: center;
            padding: 1.2rem;
            border-radius: 16px;
            margin: 0.8rem 0;
            border: 2px solid;
        }
        .result-cat { background: #fff4e6; border-color: #ff922b; color: #7c4a03; }
        .result-dog { background: #e7f5ff; border-color: #339af0; color: #0b3d66; }
        .result-card h1 { margin: 0; font-size: 2.4rem; }
        .result-card p  { margin: 0.3rem 0 0 0; font-size: 1.05rem; }

        .footer {
            text-align: center; color: #888; font-size: 0.85rem; margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model loading (cached so it only happens once per server session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


if not MODEL_PATH.exists():
    st.error(
        f"Model file `{MODEL_PATH.name}` was not found next to `app.py`. "
        "Download `best_efficientnet.keras` from Colab and place it in the same "
        "folder as this script."
    )
    st.stop()

model = load_model()

IMG_SIZE = tuple(int(s) for s in model.input_shape[1:3])
TOTAL_PARAMS = int(model.count_params())
TRAINABLE_PARAMS = int(sum(np.prod(w.shape) for w in model.trainable_weights))


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)


@st.cache_data(show_spinner=False)
def predict_dog_probability(file_bytes: bytes) -> float:
    """Return P(dog) in [0, 1]. Cached per image so slider changes are instant."""
    x = preprocess_image(Image.open(io.BytesIO(file_bytes)))
    return float(model.predict(x, verbose=0)[0][0])


def classify(p_dog: float, threshold: float):
    """Turn the sigmoid output into (label, confidence, p_cat, p_dog)."""
    p_cat = 1.0 - p_dog
    label = "Dog" if p_dog >= threshold else "Cat"
    confidence = p_dog if label == "Dog" else p_cat
    return label, confidence, p_cat, p_dog


def show_image(img: Image.Image, caption: str) -> None:
    """Show an image full-width (works on old and new Streamlit versions)."""
    try:
        st.image(img, caption=caption, width="stretch")
    except Exception:
        st.image(img, caption=caption, use_container_width=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Decision threshold",
        0.05, 0.95, 0.50, 0.05,
        help="The model outputs P(dog). Images with P(dog) at or above this "
             "value are labelled Dog; below it, Cat. 0.50 is the standard choice.",
    )
    low_conf_pct = st.slider(
        "Warn if confidence is below (%)",
        50, 95, 70,
        help="Show a warning when the model is not very sure about its answer.",
    )
    show_chart = st.checkbox("Show Cat vs Dog probability chart", value=True)

    st.divider()
    st.markdown("**Quick facts**")
    st.markdown(
        "- Model: EfficientNetB0 (custom-trained)\n"
        f"- Input: {IMG_SIZE[0]} × {IMG_SIZE[1]} RGB\n"
        "- Output: P(dog), sigmoid\n"
        f"- Parameters: {TOTAL_PARAMS / 1e6:.2f} M"
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🐱 Cat vs Dog Classifier 🐕</h1>
        <p>Upload a photo and my custom-trained EfficientNetB0 model will tell you
        whether it shows a cat or a dog.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_classify, tab_project, tab_model = st.tabs(
    ["🔍 Classifier", "📘 About Project", "🧠 About Model"]
)

# ============================================================================
# TAB 1: Classifier
# ============================================================================

with tab_classify:
    uploaded_file = st.file_uploader(
        "📤 Choose an image (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started.")
    else:
        file_bytes = uploaded_file.getvalue()
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes)))

        # Centered preview
        left, center, right = st.columns([1, 2, 1])
        with center:
            show_image(image, "Uploaded image")

        # Predict
        with st.spinner("🔍 Analyzing image..."):
            p_dog_raw = predict_dog_probability(file_bytes)
        label, confidence, cat_prob, dog_prob = classify(p_dog_raw, threshold)

        # Final decision card
        emoji = "🐱" if label == "Cat" else "🐕"
        css_class = "result-cat" if label == "Cat" else "result-dog"
        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <h1>{emoji} It's a {label}!</h1>
                <p>Confidence: <b>{confidence * 100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence
        st.progress(min(max(confidence, 0.0), 1.0))
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{confidence * 100:.1f}%")
        m2.metric("🐱 Cat probability", f"{cat_prob * 100:.1f}%")
        m3.metric("🐕 Dog probability", f"{dog_prob * 100:.1f}%")

        if confidence * 100 < low_conf_pct:
            st.warning(
                "⚠️ The model is not very sure about this one. Try a clearer, "
                "well-lit photo with a single animal in the frame."
            )

        if show_chart:
            st.markdown("##### Cat vs Dog probability")
            chart_df = pd.DataFrame(
                {"Probability (%)": [cat_prob * 100, dog_prob * 100]},
                index=["Cat", "Dog"],
            )
            st.bar_chart(chart_df)

        st.caption(
            "ℹ️ This model only knows two classes. A photo of anything other than "
            "a cat or a dog (a person, a car...) will still be labelled as one of them."
        )

# ============================================================================
# TAB 2: About Project
# ============================================================================

with tab_project:
    st.subheader("About the Project")
    st.write(
        "This project is a **Cat vs Dog image classifier** built with "
        "**transfer learning**."
        "I trained a custom classification head on top of "
        "a pre-trained **EfficientNetB0** in Google Colab, saved the best "
        "checkpoint, and deployed it in this Streamlit app."
    )

    st.markdown("#### Objective")
    st.write(
        "Compare pre-trained CNNs on a cat/dog dataset, pick the stronger base "
        "network, train it on the project's own data and serve it as an "
        "easy-to-use web app."
    )

    st.markdown("#### How a prediction works")
    st.markdown(
        "1. **Upload** a JPG, JPEG or PNG image.\n"
        f"2. The image is orientation-corrected, converted to RGB and **resized to "
        f"{IMG_SIZE[0]} × {IMG_SIZE[1]}**.\n"
        "3. Raw pixel values (0–255) go into the model; EfficientNet's built-in "
        "layers handle the scaling and normalisation.\n"
        "4. The model outputs a single sigmoid value, **P(dog)**.\n"
        "5. If P(dog) is at or above the threshold (default 0.5) the image is "
        "labelled **Dog**, otherwise **Cat**. Confidence is the probability of "
        "the predicted class."
    )

    st.markdown("#### Training setup")
    st.markdown(
        "- **Base:** EfficientNetB0 pre-trained on ImageNet, frozen (used as a "
        "feature extractor)\n"
        "- **Head:** GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → "
        "Dense(1, sigmoid)\n"
        "- **Loss / optimizer:** binary cross-entropy, Adam\n"
        "- **Augmentation:** rotation, width/height shifts, shear, zoom and "
        "horizontal flip\n"
        "- **Callbacks:** EarlyStopping, ReduceLROnPlateau and ModelCheckpoint "
        "(best validation accuracy is saved as `best_efficientnet.keras`)"
    )

    st.markdown("#### Experiment results")
    st.write(
        "Before training, both candidate base networks were evaluated "
        "**as-is** (ImageNet weights, no training) by mapping their ImageNet "
        "classes to cat/dog. EfficientNetB0 was the stronger starting point:"
    )
    rows = [
        {"Model": "MobileNetV3Large (ImageNet, as-is)", "Accuracy": "88.75%"},
        {"Model": "EfficientNetB0 (ImageNet, as-is)", "Accuracy": "90.78%"},
    ]
    if CUSTOM_ACCURACY is not None:
        rows.append(
            {
                "Model": "EfficientNetB0 trained by Transfer Learning (Feature Extractor Method)",
                "Accuracy": f"{CUSTOM_ACCURACY * 100:.2f}%",
            }
        )
    st.table(pd.DataFrame(rows).set_index("Model"))
    st.caption(
        "The as-is accuracies were measured on a subset of the test generator's "
        "batches, not the full 5,000-image test set, so treat them as estimates."
    )

    st.markdown("#### Dataset")
    st.write(
        "20,016 training images and 5,000 test images in two classes (cats and "
        "dogs), stored as `train/` and `test/` folders. Labels follow alphabetical "
        "folder order: **cats = 0, dogs = 1**."
    )

    st.markdown("#### Tech stack")
    st.markdown(
        "- **`Python`**\n"
        "- **`TensorFlow / Keras`**: `model training and inference`\n"
        "- **`Streamlit`**: `web interface`\n"
        "- **`Pillow & NumPy`**: `image handling`\n"
        "- **`Google Colab (T4 GPU)`**: `training`"
    )

    st.markdown("#### Limitations")
    st.markdown(
        "- It is a **binary** classifier: any image, including people, cars or "
        "other animals, is forced into 'cat' or 'dog'.\n"
        "- Poor lighting, heavy cropping or several animals in one photo can "
        "reduce accuracy.\n"
        "- The confidence is the model's sigmoid output; it is not a calibrated "
        "probability and can be over-confident."
    )

# ============================================================================
# TAB 3: About Model
# ============================================================================

with tab_model:
    st.subheader("About the Model")
    st.write(
        "The app uses **EfficientNetB0** as a frozen feature extractor with a "
        "small classifier head trained for cats vs dogs. "
        "**EfficientNet** is a family of convolutional neural networks introduced "
        "by Google researchers (Tan & Le, 2019); **B0** is the smallest baseline "
        "model, from which B1 to B7 are scaled up."
    )

    st.markdown("#### Key idea: compound scaling")
    st.write(
        "Instead of making a network deeper, wider, or fed with larger images "
        "independently, EfficientNet scales **depth, width and input resolution "
        "together** using a single compound coefficient. This gives better "
        "accuracy for the same compute budget."
    )

    st.markdown("#### Architecture of this model")
    feat = IMG_SIZE[0] // 32
    layers_df = pd.DataFrame(
        {
            "Layer": [
                "EfficientNetB0 (frozen, ImageNet weights)",
                "GlobalAveragePooling2D",
                "Dense (128, ReLU)",
                "Dropout (0.3)",
                "Dense (1, sigmoid)",
            ],
            "Output shape": [
                f"({feat}, {feat}, 1280)",
                "(1280)",
                "(128)",
                "(128)",
                "(1)",
            ],
            "Role": [
                "Extracts visual features (edges, fur texture, ears, eyes...)",
                "Collapses each feature map to one number",
                "Learns cat/dog-specific combinations of the features",
                "Randomly drops units during training to reduce overfitting",
                "Outputs P(dog); P(cat) = 1 - P(dog)",
            ],
        }
    )
    st.table(layers_df.set_index("Layer"))

    st.markdown("#### Model specifications")
    specs = pd.DataFrame(
        {
            "Property": [
                "Input size",
                "Input pixel range",
                "Output",
                "Class labels",
                "Total parameters",
                "Trainable parameters (head)",
                "Frozen parameters (EfficientNetB0 base)",
                "Base pre-training",
                "Framework",
            ],
            "Value": [
                f"{IMG_SIZE[0]} × {IMG_SIZE[1]} × 3",
                "Raw 0–255 (normalised inside the model)",
                "1 sigmoid unit = P(dog)",
                "0 = Cat, 1 = Dog",
                f"{TOTAL_PARAMS:,}",
                f"{TRAINABLE_PARAMS:,}",
                f"{TOTAL_PARAMS - TRAINABLE_PARAMS:,}",
                "ImageNet-1k (≈ 1.28M images)",
                "TensorFlow / Keras 3",
            ],
        }
    )
    st.table(specs.set_index("Property"))

    st.markdown("#### What is transfer learning?")
    st.write(
        "A network trained on ImageNet has already learned general visual "
        "features: edges, textures, fur patterns, shapes of ears and eyes. "
        "Rather than relearning those from scratch, the EfficientNetB0 base is "
        "kept frozen and only the small head (about "
        f"{TRAINABLE_PARAMS / 1e3:.0f}K parameters) is trained on the cat/dog "
        "dataset. This trains quickly and works well even with a modest amount "
        "of data."
    )

    st.markdown("#### Custom head vs mapping ImageNet classes")
    st.write(
        "An earlier version of this app used the stock ImageNet model and "
        "summed the probabilities of the 10 cat classes and the 118 dog classes "
        "to decide between cat and dog. The custom-trained model replaces that "
        "workaround: it learns the cat/dog decision **directly** from this "
        "project's data and returns one probability."
    )

    st.markdown("#### Why EfficientNetB0 over MobileNetV3Large?")
    st.write(
        "In the as-is comparison, EfficientNetB0 scored higher "
        "(99.16% vs 99.06%) on the evaluated test images, so it was chosen as "
        "the base network. MobileNetV3Large is lighter and faster, which makes "
        "it a good choice for mobile or edge devices, while EfficientNetB0 "
        "trades a little speed for accuracy."
    )

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------

st.markdown(
    f"<div class='footer'>Built with Streamlit & TensorFlow • "
    f"Custom-trained EfficientNetB0 • Made by {AUTHOR}</div>",
    unsafe_allow_html=True,
)