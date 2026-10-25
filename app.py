from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Nigerian Pidgin Research Lab', page_icon='🔬', layout='wide')
BASE = Path(__file__).resolve().parent
SPLITS = {'Training': 'train.tsv', 'Development': 'dev.tsv', 'Test': 'test.tsv'}
LABELS = ['negative', 'neutral', 'positive']

@st.cache_data
def load_split(filename):
    path = BASE / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, sep='\t', dtype=str)
    if not {'tweet', 'label'}.issubset(df.columns):
        raise ValueError(f'{filename} must contain tweet and label columns.')
    return df[['tweet', 'label']].dropna().copy()

st.sidebar.title('🔬 Research Navigation')
section = st.sidebar.radio('Explore My Research', ['Research Overview', 'Dataset Exploration', 'Model Experiments', 'Research Findings', 'Future Research'])
st.sidebar.divider()
st.sidebar.caption('Nigerian Pidgin NLP Research Lab')

if section == 'Research Overview':
    st.title('🔬 Nigerian Pidgin Research Lab')
    st.subheader('Sentiment Analysis Using Machine Learning and Transformer-Based Models')
    st.caption("Built by Chichi Okafor as part of my Data Science internship at DataraFlow.")
    st.write('An interactive research portfolio exploring Nigerian Pidgin sentiment datasets, machine-learning experiments, and research findings.')
    st.info('Explore the Dataset Exploration section to search real labelled tweets.')
    st.subheader("About this dashboard")
    st.write("Explore Nigerian Pidgin sentiment distributions, search tweets by keyword and sentiment label, and review preliminary results from Logistic Regression, Linear SVM, and AfriBERTa Base.")
    st.caption("Tweet labels displayed in Dataset Exploration come from the uploaded dataset; they are not live model predictions.")
    st.subheader("Dataset source")
    st.write("This dashboard explores the Nigerian Pidgin portion of NaijaSenti using the training, development, and test TSV files included in this project.")
    st.markdown("**Reference:** Muhammad et al. (2022). *NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis.*")
    st.link_button("Read the NaijaSenti paper", "https://aclanthology.org/2022.lrec-1.63/")
    st.subheader("Limitations and ongoing work")
    st.write("The training data contains relatively few neutral examples. Code-mixing and spelling variation are additional challenges being investigated.")
    st.write("Model results are preliminary. Consistent evaluation across models and further class-level analysis are needed before drawing firm conclusions.")
elif section == 'Dataset Exploration':
    st.title('📊 Dataset Exploration')
    datasets = {name: load_split(file) for name, file in SPLITS.items()}
    missing = [name for name, df in datasets.items() if df is None]
    if missing:
        st.error('Dataset files not found: ' + ', '.join(missing) + '. Upload train.tsv, dev.tsv and test.tsv into the data/ folder in the GitHub repository.')
    else:
        counts = pd.DataFrame({name: df['label'].value_counts().reindex(LABELS, fill_value=0) for name, df in datasets.items()}).T
        counts.index.name = 'Split'
        st.subheader('Sentiment distribution')
        st.dataframe(counts.rename(columns=str.title), use_container_width=True)
        st.bar_chart(counts)
        st.caption('Counts calculated directly from the uploaded NaijaSenti Nigerian Pidgin TSV files.')
        st.subheader('Explore actual Nigerian Pidgin tweets')
        split = st.selectbox('Choose dataset', list(SPLITS))
        sentiment = st.selectbox('Sentiment label', ['All', 'negative', 'neutral', 'positive'])
        keyword = st.text_input('Search for a word or phrase (optional)', placeholder='Type a Pidgin word or phrase')
        filtered = datasets[split]
        if sentiment != 'All':
            filtered = filtered[filtered['label'] == sentiment]
        if keyword.strip():
            filtered = filtered[filtered['tweet'].str.contains(keyword.strip(), case=False, regex=False, na=False)]
        st.metric('Matching tweets', f'{len(filtered):,}')
        st.dataframe(filtered.reset_index(drop=True), hide_index=True, use_container_width=True)
        st.download_button('Download filtered tweets (CSV)', data=filtered.to_csv(index=False).encode('utf-8'), file_name='filtered_pidgin_tweets.csv', mime='text/csv')
        st.subheader('Class imbalance')
        st.write('The training set contains 72 neutral examples out of 5,121 (1.4%). This imbalance may contribute to difficulties identifying neutral sentiment, but further experiments are needed to establish its effect.')
        st.write('Other linguistic challenges include code-mixing and orthographic variation.')

elif section == 'Model Experiments':
    st.title("Model Experiments")
    st.subheader("Preliminary development-set results")

    scores = pd.DataFrame({
        "Model": ["Logistic Regression", "Linear SVM"],
        "Macro-F1": [0.429883, 0.413177]
    })

    st.dataframe(scores, hide_index=True, use_container_width=True)
    st.bar_chart(
        scores.set_index("Model"),
        y="Macro-F1",
        horizontal=True
    )

    st.write("Both baselines use TF-IDF features and balanced class weights. They were trained on 3,469 cleaned training examples and evaluated on 1,281 development examples.")
    st.caption("Dataset Exploration displays the original uploaded splits. The model training data was filtered separately to remove overlap with evaluation texts, conflicting labels, and duplicate tweets.")

    st.write("In these baseline experiments, neither model predicted the neutral class on the development set.")
    st.info("AfriBERTa Base: model and training setup prepared. Evaluation results are not included in this dashboard yet.")
    st.caption("These are preliminary development results, not final test-set results.")
    
    
    
    
    
    


elif section == 'Research Findings':
    st.title('🧠 Research Findings')
    st.write('Initial experiments highlight challenges in sentiment classification, particularly the neutral class, alongside code-mixing and orthographic variation.')
    st.write('These findings are preliminary and require further evaluation.')

elif section == 'Future Research':
    st.title('🚀 Future Research')
    st.write('Next steps include consistent evaluation across models, language-aware preprocessing and further investigation of multilingual transformer models.')

st.divider()
st.link_button('View My Research on GitHub', 'https://github.com/chichiokafor/Sentiment-Analysis-of-Nigerian-Pidgin-Using-Machine-Learning-and-Transformer-Based-Models-')
