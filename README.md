#  YouTube Comments Sentiment Analysis App

An interactive **Streamlit web application** that fetches live YouTube comments using the **Google YouTube Data API**, performs **sentiment analysis** using VADER NLP, and visualizes audience opinions through dynamic charts and word clouds.

---

## 🚀 Features

- 🎥 **Real-time comment ingestion** from any public YouTube video (up to 5,000 comments)
- 💬 **Sentiment analysis** (Positive / Neutral / Negative) using VADER
- 📊 **Visual dashboards**: bar chart, pie chart, and trend line for engagement insights
- ☁️ **Deployed via Streamlit Cloud**
- 🌐 **Interactive filtering** by language and date range
- 🧠 **Word clouds** for positive and negative sentiment keywords
- 📁 **Downloadable CSV** for external analysis or reporting

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas |
| **Visualization** | Plotly, Matplotlib, WordCloud |
| **Sentiment Analysis** | VADER (NLTK) |
| **APIs** | Google YouTube Data API v3 |
| **Environment & Secrets** | python-dotenv, Streamlit Cloud Secrets |
| **Deployment** | Streamlit Cloud |

---

## 🧠 Project Workflow

```mermaid
graph TD
    A[YouTube API] -->|Fetches comments| B[Raw Comments DataFrame]
    B --> C[Preprocessing & Cleaning]
    C --> D[Sentiment Analysis (VADER)]
    D --> E[Visualizations & KPIs in Streamlit]
    E --> F[Downloadable Reports & Wordclouds]

