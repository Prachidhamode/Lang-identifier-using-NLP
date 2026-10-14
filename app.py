import streamlit as st
import pandas as pd
import unicodedata

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Language Identification",
    page_icon="🌐",
    layout="centered"
)


# Detect writing system
def detect_script(text):
    script_counts = {
        "Latin": 0,
        "Devanagari": 0,
        "Bengali": 0,
        "Gujarati": 0,
        "Tamil": 0,
        "Telugu": 0
    }

    for char in text:
        if not char.isalpha():
            continue

        name = unicodedata.name(char, "")

        for script in script_counts:
            if script.upper() in name:
                script_counts[script] += 1
                break

    present = [
        script for script, count in script_counts.items()
        if count > 0
    ]

    if len(present) == 0:
        return "Unsupported Language"

    if "Latin" in present and len(present) > 1:
        return "Mixed-language text"

    if "Latin" in present:
        return "Unsupported Language"

    if len(present) > 1:
        return "Mixed-language text"

    supported_scripts = {
        "Devanagari",
        "Bengali",
        "Gujarati",
        "Tamil",
        "Telugu"
    }

    if present[0] not in supported_scripts:
        return "Unsupported Language"

    return None


# Train model
@st.cache_resource
def train_model():
    df = pd.read_csv(
        "indian_languages.csv",
        encoding="utf-8-sig"
    )

    df = df.dropna(subset=["Text", "Language"])
    df["Text"] = df["Text"].astype(str)
    df["Language"] = df["Language"].astype(str).str.strip()

    X = df["Text"]
    y = df["Language"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 3)
            )
        ),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, len(df), sorted(y.unique())


model, accuracy, total_rows, languages = train_model()


# Web interface
st.title("🌐 Language Identification Using NLP")

st.write(
    "Enter a sentence to identify its language. "
    "The model supports Hindi, Marathi, Bengali, "
    "Gujarati, Tamil, and Telugu."
)

st.info(
    "Supported languages: Hindi, Marathi, Bengali, "
    "Gujarati, Tamil, and Telugu."
)

user_text = st.text_area(
    "Enter your sentence:",
    placeholder="Example: मला मराठी भाषा आवडते.",
    height=120
)


# Predict language
if st.button("Identify Language", type="primary"):

    text = user_text.strip()

    if not text:
        st.warning("Please enter a sentence first.")

    else:
        script_result = detect_script(text)

        if script_result == "Unsupported Language":
            st.warning("Unsupported Language")
            st.write(
                "This app currently supports only "
                "Hindi, Marathi, Bengali, Gujarati, "
                "Tamil, and Telugu."
            )

        elif script_result == "Mixed-language text":
            st.warning("Mixed-language text detected.")
            st.write(
                "Please enter a sentence written in "
                "one supported writing system."
            )

        else:
            prediction = model.predict([text])[0]

            st.success(
                f"Predicted Language: {prediction}"
            )

            probabilities = model.predict_proba([text])[0]
            confidence = max(probabilities) * 100

            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is an estimate, not a guarantee "
                "that the prediction is correct."
            )


# Sidebar
st.sidebar.header("About the Project")

st.sidebar.write(
    "This project uses Natural Language Processing, "
    "character-level TF-IDF, and Multinomial Naive Bayes."
)

st.sidebar.metric("Dataset sentences", total_rows)
st.sidebar.metric("Supported languages", len(languages))
st.sidebar.metric(
    "Test accuracy",
    f"{accuracy * 100:.2f}%"
)

st.sidebar.caption(
    "Test accuracy is measured on a held-out portion "
    "of the dataset."
)

st.divider()

st.caption(
    "Mini Project | Language Identification Using NLP"
)
