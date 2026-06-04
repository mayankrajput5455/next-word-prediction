# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------
# # Page Config
# # ------------------------------
# st.set_page_config(
#     page_title="Next Word Prediction",
#     page_icon="🧠",
#     layout="centered"
# )

# # ------------------------------
# # Load Model + Files
# # ------------------------------
# @st.cache_resource
# def load_resources():
#     model = load_model("lstm_model (1).h5")

#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)

#     with open("max_len.pkl", "rb") as f:
#         max_len = pickle.load(f)

#     return model, tokenizer, max_len


# model, tokenizer, max_len = load_resources()

# # ------------------------------
# # Prediction Function
# # ------------------------------
# def predict_next_word(text):
#     sequence = tokenizer.texts_to_sequences([text])[0]

#     sequence = pad_sequences(
#         [sequence],
#         maxlen=max_len - 1,
#         padding='pre'
#     )

#     preds = model.predict(sequence, verbose=0)

#     predicted_index = np.argmax(preds)

#     for word, index in tokenizer.word_index.items():
#         if index == predicted_index:
#             return word

#     return "No prediction"

# # ------------------------------
# # UI
# # ------------------------------
# st.title("🧠 Next Word Prediction (LSTM)")
# st.write(
#     "Enter a sentence and the model will predict the next word."
# )

# user_input = st.text_input(
#     "✍️ Enter text:",
#     placeholder="Type a sentence here..."
# )

# if st.button("Predict Next Word"):

#     if user_input.strip() == "":
#         st.warning("Please enter some text.")

#     else:
#         try:
#             next_word = predict_next_word(user_input)

#             st.success(
#                 f"Predicted Next Word: **{next_word}**"
#             )

#         except Exception as e:
#             st.error(f"Error: {e}")

# # ------------------------------
# # Footer
# # ------------------------------
# st.markdown("---")
# st.caption("LSTM-based Next Word Prediction using Streamlit")

import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
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

# Reduce TensorFlow logs
tf.get_logger().setLevel("ERROR")

# ------------------------------
# Load Resources
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        model = load_model("lstm_model (1).h5", compile=False)

        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)

        # Create reverse mapping once
        index_to_word = {
            index: word
            for word, index in tokenizer.word_index.items()
        }

        return model, tokenizer, max_len, index_to_word

    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.stop()

model, tokenizer, max_len, index_to_word = load_resources()

# ------------------------------
# Prediction Function
# ------------------------------
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return "Unknown Input"

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    predicted_index = int(np.argmax(prediction))

    return index_to_word.get(
        predicted_index,
        "No Prediction"
    )

# ------------------------------
# UI
# ------------------------------
st.title("🧠 Next Word Prediction (LSTM)")
st.write(
    "Enter a sentence and the model will predict the next word."
)

user_input = st.text_input(
    "✍️ Enter text:",
    placeholder="Type a sentence here..."
)

if st.button("Predict Next Word"):

    if not user_input.strip():
        st.warning("Please enter some text.")

    else:
        with st.spinner("Predicting..."):
            try:
                result = predict_next_word(user_input)

                st.success(
                    f"Predicted Next Word: **{result}**"
                )

            except Exception as e:
                st.error(
                    f"Prediction failed: {str(e)}"
                )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption(
    "LSTM-based Next Word Prediction using Streamlit"
)
