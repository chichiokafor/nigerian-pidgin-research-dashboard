
import streamlit as st

st.set_page_config(
    page_title="Nigerian Pidgin Research Lab",
    page_icon="🔬",
    layout="wide"
)

st.sidebar.title("🔬 Research Navigation")

section = st.sidebar.radio(
    "Explore My Research",
    [
        "Research Overview",
        "Dataset Exploration",
        "Model Experiments",
        "Research Findings",
        "Future Research"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Nigerian Pidgin NLP Research Lab")


st.title("🔬 Nigerian Pidgin Research Lab")
st.subheader("Sentiment Analysis Using Machine Learning and Transformer-Based Models")

st.write(
    "An interactive research portfolio documenting my journey "
    "through Natural Language Processing, dataset exploration, "
    "machine-learning experiments and model evaluation."
)

st.divider()

st.header("📖 About My Research")
st.write(
    "This research investigates sentiment classification in Nigerian "
    "Pidgin using traditional machine-learning methods and "
    "transformer-based language models."
)

st.header("📊 Initial Experimental Results")

st.dataframe(
    {
        "Model": [
            "Logistic Regression",
            "Linear SVM",
            "AfriBERTa Base"
        ],
        "Macro-F1": [0.4309, 0.4134, 0.4154]
    },
    hide_index=True
)

st.caption(
    "These are initial experimental results. Evaluation conditions "
    "must be aligned before making definitive model comparisons."
)

st.header("🧠 Research Challenges")
st.write(
    "Key challenges include code-mixing, orthographic variation, "
    "class imbalance and difficulties identifying neutral sentiment."
)

st.header("🚀 Future Research")
st.write(
    "Future work will investigate improved evaluation, "
    "language-aware preprocessing and additional multilingual models."
)

st.header("🔗 Research Repository")
st.link_button(
    "View My Research on GitHub",
    "https://github.com/chichiokafor/"
    "Sentiment-Analysis-of-Nigerian-Pidgin-Using-Machine-Learning-and-Transformer-Based-Models-"
)
