# 🐱 Cat vs Dog Image Classifier 🐕
 
A Streamlit web app that classifies an uploaded image as a **Cat** or **Dog** using an
**EfficientNetB0 model trained in Google Colab** (`best_efficientnet.keras`).
 
The model is EfficientNetB0 (frozen, ImageNet weights) + GlobalAveragePooling → Dense(128, ReLU)
→ Dropout(0.3) → Dense(1, sigmoid). It outputs `P(dog)`; class 0 = cat, class 1 = dog.
 
## Project structure
 
```
.
├── app.py                    # Streamlit application
├── best_efficientnet.keras   # Trained model from Colab
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme and upload-size settings
├── .gitignore
└── README.md
```
 
## Run locally
 
Use **Python 3.11 or 3.12** (TensorFlow can crash on very new Python versions).
 
```bash
python -m venv .venv
source .venv/bin/activate      
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
 
## Deploy on Streamlit Community Cloud
 
1. Push this folder to GitHub, **including `best_efficientnet.keras`** (about 18 MB, well under GitHub's 100 MB limit).
2. Go to https://share.streamlit.io and click **New app**.
3. Select the repo and branch, and set the main file path to `app.py`.
4. Under **Advanced settings**, choose Python **3.11** or **3.12**.
5. Deploy.
## Important notes
 
- **Pixel scaling:** the model expects raw 0–255 pixels (EfficientNet normalises internally). Do not
  divide by 255 or call `preprocess_input`, otherwise predictions collapse to roughly the same value.
- **Input size:** read automatically from the model (currently 224 × 224).
- **Versions:** the model was saved with Keras 3.13, so use TensorFlow 2.20+ and Keras ≥ 3.13.
- **Binary model:** any image (people, cars, other animals) is forced into cat or dog. Low-confidence
  results trigger a warning.
- Optional: set `CUSTOM_ACCURACY` at the top of `app.py` (`0.9916`) to show your Colab test
  accuracy in the *About Project* tab.
