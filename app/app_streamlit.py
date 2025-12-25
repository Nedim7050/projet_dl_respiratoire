import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

IMG_SIZE = (224, 224)


@st.cache_resource
def load_my_model():
    model = load_model("best_model.h5")
    return model


@st.cache_resource
def load_class_indices():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # Inverse : index -> nom de classe
    idx_to_class = {v: k for k, v in class_indices.items()}
    return idx_to_class


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch


def main():
    st.title("🫁 Détection de maladies respiratoires sur radiographies pulmonaires")
    st.write(
        "Ce projet utilise un modèle Deep Learning (EfficientNetB0, Transfer Learning) "
        "pour classer une radio dans les classes : **COVID19**, **NORMAL**, **PNEUMONIA**."
    )

    model = load_my_model()
    idx_to_class = load_class_indices()

    uploaded_file = st.file_uploader(
        "📤 Chargez une radiographie (format PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_container_width=True)

        if st.button("🔍 Lancer la prédiction"):
            img_batch = preprocess_image(image)
            preds = model.predict(img_batch)[0]

            pred_idx = int(np.argmax(preds))
            class_name = idx_to_class[pred_idx]
            proba = float(preds[pred_idx])

            st.subheader(f"Résultat : **{class_name}**")
            st.write(f"Probabilité associée : **{proba * 100:.2f}%**")

            st.write("Détails des probabilités par classe :")
            for i, p in enumerate(preds):
                st.write(f"- {idx_to_class[i]} : {p * 100:.2f} %")


if __name__ == "__main__":
    main()
