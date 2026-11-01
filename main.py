#!/usr/bin/env python3
"""
GeoPredict — End-to-End Intelligence Pipeline Runner
Executes the full pipeline:
  1. Scrape BBC News (Business & World News)
  2. Parse HTML cards into CSV
  3. RoBERTa Transformer Sentiment Analysis
  4. Market Relevance & LangChain Gemini Filtering
  5. Dispatch Email Alerts
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env
ENV_FILE = REPO_ROOT / "notebooks" / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

from WEB_SCRAPED.scra import scrape_bbc_headlines
from WEB_SCRAPED.collect import process_html_files, save_to_csv
from models.roberta_sentiment import RobertaSentimentAnalyzer
from models.model import filter_alerts, send_email, format_alert_email_content
from models.llm_filter import get_gemini_api_key


def run_pipeline(
    skip_scrape: bool = False,
    dry_run: bool = False,
    threshold: float = 0.70,
    mode: str = None,
    skip_llm: bool = False
):
    print("=" * 65)
    print("       🌍 GEOPREDICT INTELLIGENCE & ALERT PIPELINE")
    print("=" * 65)

    mode = mode or os.getenv("SENTIMENT_MODE", "positive")

    data_dir = REPO_ROOT / "WEB_SCRAPED" / "data"
    news_csv = REPO_ROOT / "WEB_SCRAPED" / "news_data.csv"
    alerts_csv = REPO_ROOT / "notebooks" / "rsentiments.csv"
    sentiment_csv = REPO_ROOT / "notebooks" / "sentiment.csv"

    # Stage 1: Web Scraping
    print("\n[Stage 1/5] 🌐 Scraping BBC News...")
    if skip_scrape:
        print("  ⏩ Skipped web scraping (--skip-scrape). Using existing raw data.")
    else:
        scrape_bbc_headlines(
            base_url="https://www.bbc.com/",
            sections=["business", "news"],
            output_folder=str(data_dir),
            headless=True
        )

    # Stage 2: HTML Parsing
    print("\n[Stage 2/5] 📄 Parsing HTML into news_data.csv...")
    if not skip_scrape or not news_csv.exists():
        df_news = process_html_files(str(data_dir))
        save_to_csv(df_news, str(news_csv))
    else:
        df_news = pd.read_csv(news_csv)
    print(f"  ✅ Loaded {len(df_news)} articles from {news_csv}")

    if df_news.empty:
        print("⚠️ No articles found. Pipeline exiting.")
        return

    # Stage 3: RoBERTa Sentiment Analysis
    print("\n[Stage 3/5] 🤖 Running Fine-Tuned RoBERTa Sentiment Evaluation...")
    analyzer = RobertaSentimentAnalyzer()
    texts = (df_news["Headline"].fillna("") + ". " + df_news["Description"].fillna("")).tolist()
    preds = analyzer.predict_batch(texts)

    df_news["sentiment_label"] = [p["label"] for p in preds]
    df_news["sentiment_score"] = [p["sentiment_score"] for p in preds]
    df_news["sentiment_confidence"] = [p["confidence"] for p in preds]

    pos_count = (df_news["sentiment_label"] == "Positive").sum()
    neu_count = (df_news["sentiment_label"] == "Neutral").sum()
    neg_count = (df_news["sentiment_label"] == "Negative").sum()
    print(f"  ✅ Evaluated {len(df_news)} articles:")
    print(f"     • Positive: {pos_count} | Neutral: {neu_count} | Negative: {neg_count}")

    # Stage 4: Filtering
    score_desc = f"|Score| >= {threshold:.2f}" if mode == "absolute" else f"Score >= +{threshold:.2f}"
    print(f"\n[Stage 4/5] 🎯 Filtering Alerts ({score_desc}, Topic Filter & LangChain Gemini)...")
    apply_langchain = not skip_llm and bool(get_gemini_api_key())
    if not apply_langchain and not skip_llm:
        print("  ℹ️ 'gem_key' not configured in notebooks/.env. Skipping LangChain Gemini step.")

    df_relevant = filter_alerts(
        df_news,
        min_score=threshold,
        mode=mode,
        exclude_sports=True,
        exclude_disasters=True,
        apply_langchain=apply_langchain
    )

    if not df_relevant.empty:
        if "Relevant_Headline" not in df_relevant.columns and "Headline" in df_relevant.columns:
            df_relevant["Relevant_Headline"] = df_relevant["Headline"] + " — " + df_relevant["Description"].fillna("")
        df_relevant.to_csv(alerts_csv, index=False)
        print(f"  ✅ Saved {len(df_relevant)} high-impact alerts to {alerts_csv}")

        df_relevant["timestamp"] = datetime.now().isoformat()
        df_relevant.to_csv(sentiment_csv, index=False)
        print(f"  ✅ Saved detailed records to {sentiment_csv}")
    else:
        print(f"  ℹ️ No articles passed the threshold (+{threshold}) and relevance filters.")
        # Clear old alerts CSV to prevent sending stale alerts
        pd.DataFrame().to_csv(alerts_csv, index=False)

    # Stage 5: Email Alert Dispatch
    print("\n[Stage 5/5] 📧 Sending Email Alert...")
    if df_relevant.empty:
        print("  ✅ No new alerts to dispatch.")
        return

    success = send_email(df_relevant, dry_run=dry_run)
    if success:
        print("  🎉 Pipeline completed successfully!")
    else:
        print("  ⚠️ Email dispatch failed. Please check SMTP settings in notebooks/.env.")


def main():
    parser = argparse.ArgumentParser(description="GeoPredict Global Pulse Sentiment & Alert Pipeline")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip web scraping and use existing news_data.csv")
    parser.add_argument("--dry-run", action="store_true", help="Prepare alert email without sending it via SMTP")
    parser.add_argument("--threshold", type=float, default=0.90, help="Minimum sentiment polarity score (default: 0.70)")
    parser.add_argument("--mode", choices=["positive", "absolute"], default=None, help="Filter mode: 'positive' (score >= threshold) or 'absolute' (|score| >= threshold)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LangChain Gemini filtering")
    args = parser.parse_args()

    run_pipeline(
        skip_scrape=args.skip_scrape,
        dry_run=args.dry_run,
        threshold=args.threshold,
        mode=args.mode,
        skip_llm=args.skip_llm
    )


if __name__ == "__main__":
    main()
