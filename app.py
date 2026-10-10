import streamlit as st
import pandas as pd

st.set_page_config(page_title="Nigerian Pidgin Research Lab", page_icon="🔬", layout="wide")

st.sidebar.title("🔬 Research Navigation")
section = st.sidebar.radio(
    "Explore My Research",
    ["Research Overview", "Dataset Exploration", "Model Experiments", "Research Findings", "Future Research"],
)
st.sidebar.divider()
st.sidebar.caption("Nigerian Pidgin NLP Research Lab")

if section == "Research Overview":
    st.title("🔬 Nigerian Pidgin Research Lab")
    st.subheader("Sentiment Analysis of Nigerian Pidgin Using Machine Learning and Transformer-Based Models")
    st.write("An interactive research portfolio documenting Nigerian Pidgin sentiment analysis, dataset exploration, model experiments and evaluation.")
    st.header("📖 About My Research")
    st.write("This research investigates positive, negative and neutral sentiment classification using traditional machine-learning classifiers and transformer-based language models.")

elif section == "Dataset Exploration":
    st.title("📊 Dataset Exploration")
    st.write("NaijaSenti Nigerian Pidgin dataset splits:")
    counts = pd.DataFrame({
        "Split": ["Training", "Development", "Test"],
        "Negative": [3241, 813, 2326],
        "Positive": [1808, 447, 1397],
        "Neutral": [72, 21, 431],
    })
    counts["Total"] = counts[["Negative", "Positive", "Neutral"]].sum(axis=1)
    st.dataframe(counts, hide_index=True)
    st.subheader("Sentiment distribution by dataset split")
    st.bar_chart(counts.set_index("Split")[["Negative", "Positive", "Neutral"]])
    st.caption("Counts calculated from the uploaded NaijaSenti Nigerian Pidgin train.tsv, dev.tsv and test.tsv files.")
    st.subheader("Class imbalance")
    st.write("The training set contains only 72 neutral examples out of 5,121 (1.4%). This pronounced class imbalance may contribute to difficulties identifying neutral sentiment, but further experiments are needed to establish its effect.")
    st.write("Other linguistic challenges include code-mixing and orthographic variation.")

elif section == "Model Experiments":
    st.title("🧪 Model Experiments")
    st.dataframe({"Model": ["Logistic Regression", "Linear SVM", "AfriBERTa Base"], "Macro-F1": [0.4309, 0.4134, 0.4154]}, hide_index=True)
    st.caption("Preliminary results only. Evaluation conditions must be aligned before definitive comparisons.")

elif section == "Research Findings":
    st.title("🧠 Research Findings")
    st.write("Initial experiments highlight challenges in sentiment classification, particularly the neutral class, alongside code-mixing and orthographic variation.")
    st.write("These findings are preliminary and require further evaluation.")

elif section == "Future Research":
    st.title("🚀 Future Research")
    st.write("Next steps include consistent evaluation across models, language-aware preprocessing and further investigation of multilingual transformer models.")

st.divider()
st.link_button("View My Research on GitHub", "https://github.com/chichiokafor/Sentiment-Analysis-of-Nigerian-Pidgin-Using-Machine-Learning-and-Transformer-Based-Models-")
