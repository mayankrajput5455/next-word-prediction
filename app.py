import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

# ------------------------------
# Load Resources
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    return model, tokenizer, max_len, index_to_word

try:
    model, tokenizer, max_len, index_to_word = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ------------------------------
# Prediction Function
# ------------------------------
def predict_next_word(text):
    text = text.lower().strip()

    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return None, None

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    # Handle both indexing styles
    word = index_to_word.get(predicted_index)

    if word is None:
        word = index_to_word.get(predicted_index + 1)

    confidence = float(np.max(prediction) * 100)

    return word, confidence

# ------------------------------
# UI
# ------------------------------
st.title("🧠 Next Word Prediction using LSTM")

st.write(
    "Enter a sentence and the model will predict the most likely next word."
)

with st.form("prediction_form"):
    user_input = st.text_input(
        "✍️ Enter text:",
        placeholder="Type a sentence here..."
    )

    submitted = st.form_submit_button("Predict Next Word")

if submitted:

    if not user_input.strip():
        st.warning("Please enter some text.")

    else:
        with st.spinner("Predicting..."):
            try:
                word, confidence = predict_next_word(user_input)

                if word:
                    st.success(
                        f"Predicted Next Word: **{word}**"
                    )

                    st.info(
                        f"Confidence: **{confidence:.2f}%**"
                    )
                else:
                    st.warning(
                        "The model could not recognize the input words."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")
