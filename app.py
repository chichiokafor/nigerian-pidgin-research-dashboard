import streamlit as st

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
    st.dataframe({"Split": ["Training", "Development", "Test"], "Examples": [5121, 1281, 4154]}, hide_index=True)
    st.write("Linguistic challenges include code-mixing and orthographic variation.")

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
