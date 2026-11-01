# 🌍 GeoPredict — Global Pulse Alert System 🌐

GeoPredict is an AI-driven sentinel that scans the global news landscape to detect geopolitical and macroeconomic events with the potential to ripple through financial markets. Rather than chasing raw price predictions, GeoPredict acts as your vigilant watchtower, sifting through the noise of world events to deliver timely, actionable alerts.

---

## 🎯 Core Pipeline Architecture

1. **🌐 Web Surveillance (`WEB_SCRAPED/scra.py`)**: Headless browser scraping of live geopolitical and economic news from BBC `/business/` and `/news/`.
2. **📄 Ingestion & Normalization (`WEB_SCRAPED/collect.py`)**: Extracts headlines, descriptions, timestamps, and tags into structured datasets (`news_data.csv`).
3. **🧠 RoBERTa Transformer Sentiment (`models/roberta_sentiment.py`)**: Fine-tuned on the Financial PhraseBank benchmark to produce discrete sentiment (`Positive`, `Neutral`, `Negative`) and continuous compound polarity scores in `[-1.0, 1.0]`.
4. **🎯 Noise Reduction & LangChain Gemini Relevance (`models/llm_filter.py`)**: Filters out sports and natural disaster noise, requires `score >= +0.70`, and prompts Google Gemini to evaluate market impact and produce qualitative reasoning.
5. **📬 Email Alert Dispatch (`models/model.py`)**: Dispatches structured, actionable intelligence summaries via SMTP.

---

## 🚀 How to Run the Pipeline

### Option 1: One-Command CLI Runner (Recommended)

From the project root:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the complete end-to-end pipeline (scrape, analyze, filter, and email)
python main.py
```

#### Useful CLI Flags:
```bash
# Test without sending actual emails (dry run)
python main.py --dry-run

# Re-run analysis on already scraped news (skip web scraping)
python main.py --skip-scrape

# Set a custom sentiment threshold (e.g., 0.80)
python main.py --threshold 0.80

# Skip LangChain LLM step (sentiment + noise filters only)
python main.py --skip-llm
```

---

### Option 2: Interactive Jupyter Notebook

If you prefer exploring step-by-step:

```bash
source .venv/bin/activate
jupyter notebook notebooks/junote.ipynb
```

Step through the notebook cells:
- **Cell 1**: Environment setup and `.env` loading
- **Cell 2**: Load news data
- **Cell 3**: RoBERTa sentiment evaluation
- **Cell 4**: LangChain Gemini market relevance filtering and CSV export
- **Cell 5**: Test set validation metrics (accuracy: 95.38%, macro F1: 0.9497)

---

## ⚙️ Environment Configuration (`notebooks/.env`)

Ensure your credentials are set in `notebooks/.env`:

```env
# Gmail SMTP settings for alert delivery
SENDER_EMAIL=your_email@gmail.com
RECEIVER_EMAIL=target_email@gmail.com
gmail="your_gmail_app_password"

# Google Gemini API Key for LangChain LLM relevance filtering
gem_key="your_gemini_api_key"
```

---

## 🧪 Running Tests

To verify that all sentiment models, topic filters, and formatting functions pass:

```bash
pytest tests/test_sentiment.py -v
```
