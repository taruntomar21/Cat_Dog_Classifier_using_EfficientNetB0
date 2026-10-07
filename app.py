import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input,
    decode_predictions,
)

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ImageNet class index ranges (inclusive)
CAT_INDICES = list(range(278, 288))  # 278 - 287  -> 10 classes
DOG_INDICES = list(range(151, 269))  # 151 - 268  -> 118 classes

AUTHOR = "Tarun"

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
@st.cache_resource(show_spinner="Loading EfficientNetB0 model...")
def load_model():
    return EfficientNetB0(weights="imagenet", include_top=True)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
    return preprocess_input(arr)


def classify(preds: np.ndarray):
    cat_prob = float(np.sum(preds[0][CAT_INDICES]))
    dog_prob = float(np.sum(preds[0][DOG_INDICES]))

    label = "Cat" if cat_prob >= dog_prob else "Dog"
    total = cat_prob + dog_prob
    confidence = (max(cat_prob, dog_prob) / total) if total > 0 else 0.0
    return label, confidence, cat_prob, dog_prob


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Number of top predictions to show", 1, 10, 3)
    min_certainty = st.slider(
        "Warn if cat+dog probability is below (%)",
        0, 100, 30,
        help="If the model puts most of its probability on non-cat/dog classes, "
             "the image is probably something else.",
    )
    show_chart = st.checkbox("Show Cat vs Dog probability chart", value=True)

    st.divider()
    st.markdown("**Quick facts**")
    st.markdown(
        "- Model: EfficientNetB0\n"
        "- Weights: ImageNet\n"
        "- Input: 224 × 224 RGB\n"
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🐱 Cat vs Dog Classifier 🐕</h1>
        <p>Upload a photo and let EfficientNetB0 tell you who's in it, and which breed it looks like.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()

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
        image = Image.open(uploaded_file)

        # Centered preview
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.image(image, caption="Uploaded image", use_container_width=True)

        # Predict
        with st.spinner("🔍 Analyzing image..."):
            x = preprocess_image(image)
            preds = model.predict(x, verbose=0)
            label, confidence, cat_prob, dog_prob = classify(preds)

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

        if (cat_prob + dog_prob) * 100 < min_certainty:
            st.warning(
                "⚠️ Most of the model's probability went to classes other than "
                "cats and dogs. This image may not contain a cat or a dog."
            )

        if show_chart:
            st.markdown("##### Cat vs Dog probability")
            chart_df = pd.DataFrame(
                {"Probability (%)": [cat_prob * 100, dog_prob * 100]},
                index=["Cat", "Dog"],
            )
            st.bar_chart(chart_df)

        # Top-N predictions
        st.markdown(f"#####  Top {top_n} predictions")
        for _, name, prob in decode_predictions(preds, top=top_n)[0]:
            st.write(f"**{name.replace('_', ' ').title()}**: {prob * 100:.2f}%")
            st.progress(float(prob))

# ============================================================================
# TAB 2: About Project
# ============================================================================
with tab_project:
    st.subheader("About the Project")
    st.write(
        "This project is a **Cat vs Dog image classifier** built with "
        "**transfer learning**. Rather than training a network from scratch, it "
        "reuses a CNN that was already trained on millions of images and adapts "
        "its output to a binary decision."
    )

    st.markdown("#### Objective")
    st.write(
        "Compare pre-trained CNNs on a cat/dog dataset, pick the best-performing "
        "one, and deploy it as an easy-to-use web app."
    )

    st.markdown("#### How it works")
    st.markdown(
        "1. **Upload** a JPG, JPEG or PNG image.\n"
        "2. The image is **resized to 224 × 224** and preprocessed with "
        "EfficientNet's `preprocess_input`.\n"
        "3. EfficientNetB0 outputs **1000 ImageNet probabilities**.\n"
        "4. Probabilities for **cat classes (indices 278–287)** are summed, and "
        "so are those for **dog classes (indices 151–268)**.\n"
        "5. The larger sum decides **Cat or Dog**; confidence is its share of the "
        "combined cat + dog probability."
    )

    st.markdown("#### Experiment results")
    st.write(
        "Both models were evaluated as-is (no fine-tuning) on images from the "
        "test set, mapping ImageNet classes to cat/dog:"
    )
    results = pd.DataFrame(
        {
            "Model": ["MobileNetV3Large", "EfficientNetB0"],
            "Images evaluated": [480, 640],
            "Accuracy": ["88.75%", "90.78%"],
        }
    )

    st.caption(
        "Accuracy was measured on a subset of the test generator's batches, not "
        "the full 5,000-image test set, so treat it as an estimate."
    )
    st.write(
        "A fine-tuned version (frozen base + small dense head trained on the "
        "dataset) was also tried, but the lightweight 100-steps-per-epoch run "
        "did not beat the pre-trained model, so the app uses the pre-trained "
        "EfficientNetB0 directly."
    )

    st.markdown("#### Dataset")
    st.write(
        "20,016 training images and 5,000 test images in two classes (cats and "
        "dogs), stored as `train/` and `test/` folders."
    )

    st.markdown("#### Tech stack")
    st.markdown(
        "- **Python**\n"
        "- **TensorFlow / Keras**: model and preprocessing\n"
        "- **Streamlit**: web interface\n"
        "- **Pillow & NumPy**: image handling\n"
        "- **Google Colab (T4 GPU)**: experimentation"
    )

    st.markdown("#### Limitations")
    st.markdown(
        "- The model only knows ImageNet's categories. Other animals or objects "
        "are forced into cat or dog unless the warning triggers.\n"
        "- Poor lighting, heavy cropping or multiple animals can reduce accuracy.\n"
        "- Confidence is relative to cat + dog only; it is not a calibrated "
        "probability."
    )

# ============================================================================
# TAB 3: About Model
# ============================================================================
with tab_model:
    st.subheader(" About EfficientNetB0")
    st.write(
        "**EfficientNet** is a family of convolutional neural networks introduced "
        "by Google researchers (Tan & Le, 2019). **B0** is the smallest baseline "
        "model of the family and the starting point from which B1 to B7 are scaled up."
    )

    st.markdown("#### Key idea: compound scaling")
    st.write(
        "Instead of making a network deeper, wider, or fed with larger images "
        "independently, EfficientNet scales **depth, width and input resolution "
        "together** using a single compound coefficient. This gives better "
        "accuracy for the same compute budget."
    )

    st.markdown("#### Architecture")
    st.markdown(
        "- Built from **MBConv blocks** (mobile inverted bottleneck convolutions).\n"
        "- Uses **squeeze-and-excitation** layers to re-weight important channels.\n"
        "- Uses the **Swish** activation.\n"
        "- Ends with global average pooling and a 1000-way softmax classifier "
        "(the `include_top=True` head used in this app)."
    )

    st.markdown("#### Model specifications")
    specs = pd.DataFrame(
        {
            "Property": [
                "Input size",
                "Parameters (with classifier head)",
                "Pre-training dataset",
                "Output classes",
                "Reported ImageNet top-1 accuracy",
                "Reported ImageNet top-5 accuracy",
                "Framework",
            ],
            "Value": [
                "224 × 224 × 3",
                "≈ 5.3 million",
                "ImageNet-1k (≈ 1.28M images)",
                "1000",
                "≈ 77%",
                "≈ 93%",
                "TensorFlow / Keras",
            ],
        }
    )
    st.table(specs.set_index("Property"))

    st.markdown("#### What is transfer learning?")
    st.write(
        "A network trained on ImageNet has already learned general visual "
        "features: edges, textures, fur patterns, shapes of ears and eyes. "
        "ImageNet also contains many dog breeds and several cat types, so the "
        "pre-trained model can recognise cats and dogs with no extra training. "
        "This app simply groups its fine-grained predictions into two buckets."
    )

    st.markdown("#### ImageNet → Cat / Dog mapping")
    mapping = pd.DataFrame(
        {
            "Group": ["🐱 Cat", "🐕 Dog"],
            "ImageNet indices": ["278 – 287", "151 – 268"],
            "Number of classes": [len(CAT_INDICES), len(DOG_INDICES)],
            "Examples": [
                "Tabby, Tiger cat, Persian, Siamese, Egyptian cat",
                "Chihuahua, Beagle, Golden retriever, Husky, Poodle",
            ],
        }
    )
    st.table(mapping.set_index("Group"))

    st.markdown("#### Why EfficientNetB0 over MobileNetV3Large?")
    st.write(
        "In the experiments for this project, EfficientNetB0 scored higher "
        "(90.78% vs 88.75%) on the evaluated test images. MobileNetV3Large is "
        "lighter and faster, which makes it a good choice for mobile or edge "
        "devices, while EfficientNetB0 trades a little speed for accuracy."
    )

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    f"<div class='footer'>Built with Streamlit & TensorFlow • "
    f"EfficientNetB0 (ImageNet weights) • Made by {AUTHOR}</div>",
    unsafe_allow_html=True,
)