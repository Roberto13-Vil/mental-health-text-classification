import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import altair as alt
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Dashboard Suicide Detection',
    page_icon=':brain:',
    layout='wide'
)

alt.theme.enable('dark')

df = pd.read_parquet('Data/data_dashboard.parquet')

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Mental Health Text Exploration</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; font-size:18px;">
    This dashboard presents an <strong>Exploratory Data Analysis (EDA)</strong> of text related to
    <span style="color:#FF6F61"><strong>suicidal</strong></span> and 
    <span style="color:#88C0D0"><strong>non-suicidal</strong></span> social media posts.
    It includes visualizations such as word distributions, word clouds, and n-gram frequencies
    to uncover linguistic patterns in mental health-related text.
</p>
""", unsafe_allow_html=True)

st.sidebar.title("🧭 Indice")
section = st.sidebar.radio("Go to section:", [
    "Max Text Stats",
    "Class Distribution",
    "WordClouds",
    "N-Grams"
])
if section == "Max Text Stats":
    st.divider()

    st.markdown("""
        <h3 style='text-align: center; color:#4B8BBE;'>📊 Max Text Stats per Class</h3>
        <p style='text-align: center; font-size:16px;'>Quick overview of text size per class (characters, tokens, unique tokens).</p>
    """, unsafe_allow_html=True)

    st.markdown("### 🟥 Suicide")

    su1, su2, su3 = st.columns(3)

    with su1:
        st.metric("🔠 Max Chars", df[df['class'] == 'suicide']['char_len'].max(), border=True)

    with su2:
        st.metric("🔤 Max Tokens", df[df['class'] == 'suicide']['token_count'].max(), border=True)

    with su3:
        st.metric("🆎 Unique Tokens", df[df['class'] == 'suicide']['unique_token'].max(), border=True)

    st.markdown("### 🟦 Non-Suicide")

    ns1, ns2, ns3 = st.columns(3)

    with ns1:
        st.metric("🔠 Max Chars", df[df['class'] == 'non-suicide']['char_len'].max(), border=True)

    with ns2:
        st.metric("🔤 Max Tokens", df[df['class'] == 'non-suicide']['token_count'].max(), border=True)

    with ns3:
        st.metric("🆎 Unique Tokens", df[df['class'] == 'non-suicide']['unique_token'].max(), border=True)

elif section == "Class Distribution":
    st.divider()

    st.markdown("""
        <h3 style='text-align: center; color: #4B8BBE;'>📊 Class Distribution</h3>
        <p style='text-align: center; font-size:16px'>✅ Dataset is perfectly balanced: 50% suicide, 50% non-suicide.<p>
    """, unsafe_allow_html=True)

    class_counts = df['class'].value_counts().reset_index()
    class_counts.columns = ['class', 'count']

    col1, _, col2 = st.columns([1.5, 0.1, 0.5])

    with col1:
        fig = px.bar(
            class_counts,
            x='class',
            y='count',
            color='class',
            color_discrete_map={
                'suicide': "#FF6F61",
                'non_suicide': "#88C0D0"
            },
            text='count',
            title="Distribution of Classes (Balanced)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            class_counts,
            values='count',
            names='class',
            color='class',
            color_discrete_map={
                'suicide': "#FF6F61",
                'non_suicide': "#88C0D0"
            },
            title="Distribution of Classes (Balanced)"
        )

        st.plotly_chart(fig, use_container_width=True)

elif section == "WordClouds":

    st.divider()

    st.markdown("""
        <div style='text-align: center;'>
            <h3 style='color: #4B8BBE;'>☁️ WordClouds</h3>
            <p style='font-size: 16px; color: #CCCCCC;'>Top words by class based on cleaned text content.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("Outputs/Images/suicide_wordcloud.png", caption='💀 Suicide', use_container_width=True)

    with col2:
        st.image("Outputs/Images/general_wordcloud.png", caption='🌐 General', use_container_width=True)

    with col3:
        st.image("Outputs/Images/non_suicide_wordcloud.png", caption='💬 Non-Suicide', use_container_width=True)

elif section == "N-Grams":

    st.divider()

    st.markdown("""
        <div style='text-align: center;'>
            <h3 style='color: #4B8BBE;'>🔠 N-Gram Frequency</h3>
            <p style='font-size: 16px; color: #CCCCCC;'>Select the N-gram size to explore the most common patterns per class.</p>
        </div>
    """, unsafe_allow_html=True)

    ngram_size = st.selectbox("🔢 Select N-gram size", [1, 2, 3], index=0, format_func=lambda x: f"{x}-gram")

    def load_ngrams(n):
        suicide_path = f'src/Data/suicide_{n}gram.pkl'
        non_suicide_path = f'src/Data/non_suicide_{n}gram.pkl'

        with open(suicide_path, 'rb') as f1, open(non_suicide_path, 'rb') as f2:
            return pickle.load(f1), pickle.load(f2)
        
    suicide_ngrams, non_suicide_ngrams = load_ngrams(ngram_size)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟥 Suicide")
        fig1 = px.bar(suicide_ngrams, x='count', y='ngram', orientation='h', text='count',
                    color_discrete_sequence=['#FF6F61'])
        
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### 🟦 Non-Suicide")
        fig2 = px.bar(non_suicide_ngrams, x='count', y='ngram', orientation='h', text='count',
                    color_discrete_sequence=['#88C0D0'])
        
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
    <hr style="margin-top: 2rem;">
    <div style="text-align: center; color: gray;">
        Built by Roberto Vilchis | Powered by Streamlit + Pandas + Plotly
    </div>
""", unsafe_allow_html=True)