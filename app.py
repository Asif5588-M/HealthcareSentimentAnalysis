import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from src.data import DataProcessor
from src.model import SentimentModel
from src.pipeline import SentimentPipeline

st.set_page_config(
    page_title="Healthcare Sentiment Analysis",
    page_icon="🏥",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.positive { background-color: #d4edda; padding: 10px; border-radius: 8px; border-left: 4px solid #28a745; }
.negative { background-color: #f8d7da; padding: 10px; border-radius: 8px; border-left: 4px solid #dc3545; }
.neutral  { background-color: #fff3cd; padding: 10px; border-radius: 8px; border-left: 4px solid #ffc107; }
.metric-card { background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def load_components():
    return DataProcessor(), SentimentModel(), SentimentPipeline()

processor, model, pipeline = load_components()

# Header
st.title("🏥 Healthcare Sentiment Analysis")
st.markdown("### AI-Powered Patient Feedback Analyzer")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Analysis",
    "📊 Bulk Analysis",
    "📈 Dashboard",
    "🕐 History"
])

# ─── TAB 1: Single Analysis ───
with tab1:
    st.subheader("📝 Analyze Single Review")

    sample_reviews = processor.get_sample_healthcare_reviews()
    use_sample = st.checkbox("Use sample review")

    if use_sample:
        text_input = st.selectbox("Select sample:", sample_reviews)
    else:
        text_input = st.text_area(
            "Enter patient review or feedback:",
            placeholder="Type healthcare review here...",
            height=150
        )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                cleaned = processor.clean_text(text_input)
                result = model.analyze_textblob(cleaned)
                pipeline.save_result(text_input, result, source="manual")

            sentiment = result['sentiment']
            css_class = sentiment.lower()

            st.markdown(f"""
            <div class="{css_class}">
                <h2>{result['emoji']} {sentiment}</h2>
                <p><b>Polarity:</b> {result['polarity']:+.3f} &nbsp;|&nbsp;
                   <b>Subjectivity:</b> {result['subjectivity']:.3f} &nbsp;|&nbsp;
                   <b>Confidence:</b> {result['confidence']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['polarity'],
                    title={'text': "Sentiment Polarity"},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': result['color']},
                        'steps': [
                            {'range': [-1, -0.1], 'color': "#ffcccc"},
                            {'range': [-0.1, 0.1], 'color': "#ffffcc"},
                            {'range': [0.1, 1], 'color': "#ccffcc"},
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['subjectivity'],
                    title={'text': "Subjectivity Score"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "#4A90D9"},
                    }
                ))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text!")

# ─── TAB 2: Bulk Analysis ───
with tab2:
    st.subheader("📊 Bulk CSV Analysis")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"]
        )
    with col2:
        use_samples = st.button(
            "📋 Use Sample Reviews",
            use_container_width=True
        )

    if use_samples:
        reviews = processor.get_sample_healthcare_reviews()
        texts = reviews
        df_bulk = pd.DataFrame({"review": texts})

        with st.spinner("Analyzing sample reviews..."):
            results = model.analyze_batch(texts)
            saved = pipeline.save_batch(
                texts, results, source="bulk_sample"
            )

        df_bulk['sentiment'] = [r['sentiment'] for r in results]
        df_bulk['emoji'] = [r['emoji'] for r in results]
        df_bulk['polarity'] = [r['polarity'] for r in results]
        df_bulk['confidence'] = [r['confidence'] for r in results]

        st.success(f"✅ {saved} reviews analyzed!")
        st.dataframe(df_bulk, use_container_width=True)

        csv = df_bulk.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            csv,
            "sentiment_results.csv",
            use_container_width=True
        )

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.write("Preview:", df_preview.head())

        text_col = st.selectbox(
            "Select text column:",
            df_preview.columns.tolist()
        )

        if st.button("🚀 Analyze CSV", use_container_width=True):
            texts = df_preview[text_col].dropna().tolist()

            with st.spinner(f"Analyzing {len(texts)} reviews..."):
                results = model.analyze_batch(texts)
                saved = pipeline.save_batch(
                    texts, results, source="csv_upload"
                )

            df_preview['sentiment'] = [r['sentiment'] for r in results]
            df_preview['polarity'] = [r['polarity'] for r in results]
            df_preview['confidence'] = [r['confidence'] for r in results]

            st.success(f"✅ {saved} reviews analyzed!")
            st.dataframe(df_preview, use_container_width=True)

            csv = df_preview.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "sentiment_results.csv",
                use_container_width=True
            )

# ─── TAB 3: Dashboard ───
with tab3:
    st.subheader("📈 Sentiment Dashboard")

    if st.button("🔄 Refresh Data"):
        st.rerun()

    history = pipeline.get_history(limit=200)
    stats = pipeline.get_stats()

    if not history:
        st.info("No data yet! Analyze some reviews first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total", stats['total'])
        with col2:
            st.metric("😊 Positive", stats['positive'])
        with col3:
            st.metric("😞 Negative", stats['negative'])
        with col4:
            st.metric("😐 Neutral", stats['neutral'])

        st.markdown("---")
        df_hist = pd.DataFrame(history)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🥧 Sentiment Distribution")
            sentiment_counts = df_hist['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    'POSITIVE': '#28a745',
                    'NEGATIVE': '#dc3545',
                    'NEUTRAL': '#ffc107'
                },
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("📊 Polarity Distribution")
            fig_hist = px.histogram(
                df_hist,
                x='polarity',
                color='sentiment',
                color_discrete_map={
                    'POSITIVE': '#28a745',
                    'NEGATIVE': '#dc3545',
                    'NEUTRAL': '#ffc107'
                },
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("☁️ Word Cloud")
        col1, col2, col3 = st.columns(3)

        sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        cols = [col1, col2, col3]
        colors = ['Greens', 'Reds', 'YlOrBr']

        for sentiment, col, color in zip(sentiments, cols, colors):
            with col:
                texts_filtered = df_hist[
                    df_hist['sentiment'] == sentiment
                ]['text'].tolist()

                if texts_filtered:
                    all_text = " ".join(texts_filtered)
                    try:
                        wc = WordCloud(
                            width=400,
                            height=200,
                            background_color='white',
                            colormap=color,
                            max_words=50
                        ).generate(all_text)

                        fig_wc, ax = plt.subplots(figsize=(5, 3))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(
                            f"{sentiment}",
                            fontsize=12,
                            fontweight='bold'
                        )
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        st.image(buf)
                        plt.close()
                    except:
                        st.info(f"Not enough {sentiment} data")
                else:
                    st.info(f"No {sentiment} reviews")

        st.subheader("📅 Sentiment Over Time")
        if 'timestamp' in df_hist.columns:
            df_hist['date'] = pd.to_datetime(
                df_hist['timestamp']
            ).dt.date
            daily = df_hist.groupby(
                ['date', 'sentiment']
            ).size().reset_index(name='count')
            fig_line = px.line(
                daily,
                x='date',
                y='count',
                color='sentiment',
                color_discrete_map={
                    'POSITIVE': '#28a745',
                    'NEGATIVE': '#dc3545',
                    'NEUTRAL': '#ffc107'
                }
            )
            st.plotly_chart(fig_line, use_container_width=True)

# ─── TAB 4: History ───
with tab4:
    st.subheader("🕐 Analysis History")

    history = pipeline.get_history(limit=50)

    if not history:
        st.info("No history yet!")
    else:
        df_hist = pd.DataFrame(history)

        if st.button("🗑️ Clear History", type="secondary"):
            pipeline.clear_history()
            st.success("History cleared!")
            st.rerun()

        st.dataframe(
            df_hist[[
                'emoji', 'sentiment', 'polarity',
                'confidence', 'source', 'timestamp', 'text'
            ]],
            use_container_width=True,
            height=400
        )

        csv = df_hist.to_csv(index=False)
        st.download_button(
            "📥 Download History",
            csv,
            "sentiment_history.csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    "**🏥 Healthcare Sentiment Analysis** | "
    "Built by Asif Nawaz | "
    "Powered by TextBlob + MongoDB + Streamlit"
)