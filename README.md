# 🐱 Cat vs Dog Image Classifier 🐕
 
A Streamlit web app that classifies an uploaded image as a **Cat** or **Dog** using a
pre-trained **EfficientNetB0** (ImageNet weights). The 1000 ImageNet class probabilities are
mapped to a binary decision:
 
- Cat: ImageNet indices 278–287
- Dog: ImageNet indices 151–268
The app also shows the confidence and the top 3 breed/class predictions.
 
## Project structure
 
```
.
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme and upload-size settings
├── .gitignore
└── README.md
```
 
## Run locally
 
```bash
# 1. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Start the app
streamlit run app.py
```
 
The first run downloads the EfficientNetB0 weights (~20 MB) automatically; no model file
needs to be saved or shipped.
 
## Deploy on Streamlit Community Cloud
 
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and click **New app**.
3. Select the repo, branch, and set the main file path to `app.py`.
4. Under **Advanced settings**, pick a Python version supported by TensorFlow (3.11 or 3.12 is a safe choice).
5. Deploy.
## Notes
 
- Model loading is cached with `@st.cache_resource`, so it happens once per server session.
- Confidence is the winning class's share of (cat probability + dog probability).
- If an image is neither a cat nor a dog, the app shows a low-certainty warning.
- On macOS with Apple Silicon, if `pip install tensorflow` fails, try `pip install tensorflow-macos`
  (older setups) or upgrade Python to a supported version.
 
