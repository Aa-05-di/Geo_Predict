import os
import re
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
import pandas as pd

from models.llm_filter import (
    get_gemini_api_key,
    get_llm_relevance,
    filter_news_with_langchain,
)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "notebooks" / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

CSV_PATH = Path(os.getenv("ALERTS_CSV_PATH", str(BASE_DIR / "notebooks" / "rsentiments.csv")))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "aadithyanlearn@gmail.com")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "maitharockers@gmail.com")
APP_PASSWORD = os.getenv("gmail")

MIN_SENTIMENT_SCORE = float(os.getenv("MIN_SENTIMENT_SCORE", "0.70"))
SENTIMENT_MODE = os.getenv("SENTIMENT_MODE", "positive")  # "positive" for score >= 0.70, "absolute" for |score| >= 0.70
EXCLUDE_SPORTS = os.getenv("EXCLUDE_SPORTS", "true").lower() in ("true", "1", "yes")
EXCLUDE_DISASTERS = os.getenv("EXCLUDE_DISASTERS", "true").lower() in ("true", "1", "yes")

SPORTS_PATTERNS = [
    r"\bsport", r"\bfootball", r"\bsoccer", r"\bcricket", r"\btennis",
    r"\bgrand slam", r"\bpremier league", r"\bchampions league", r"\bla liga",
    r"\bserie a", r"\bbundesliga", r"\bnba\b", r"\bnfl\b", r"\bfifa\b",
    r"\bman utd\b", r"\bmanchester united", r"\breal madrid", r"\bbarcelona\b",
    r"\bchelsea\b", r"\bliverpool\b", r"\barsenal\b", r"\btest match\b",
    r"\bwicket\b", r"\binnings\b", r"\bworld cup\b"
]

DISASTER_PATTERNS = [
    r"\bdisaster", r"\bearthquake", r"\bflood", r"\btsunami", r"\bhurricane",
    r"\btyphoon", r"\blandslide", r"\bavalanche", r"\bwildfire", r"\bfire\b",
    r"\bcrush\b", r"\bstampede", r"\bdeadly crash", r"\bplane crash", r"\btrain crash",
    r"\bkills?\b", r"\bkilled\b", r"\bfatalit", r"\bdeadly attack"
]


def is_blacklisted_topic(
    text: str,
    category: str = "",
    exclude_sports: bool = True,
    exclude_disasters: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Checks if an article's text or category belongs to unwanted noise categories
    (such as sports or natural/physical disasters).
    """
    combined = f"{str(text)} {str(category)}".lower()

    if exclude_sports:
        for pat in SPORTS_PATTERNS:
            if re.search(pat, combined):
                return True, f"sports ({pat})"

    if exclude_disasters:
        for pat in DISASTER_PATTERNS:
            if re.search(pat, combined):
                return True, f"disaster ({pat})"

    return False, None


def filter_alerts(
    df: pd.DataFrame,
    min_score: Optional[float] = None,
    mode: Optional[str] = None,
    exclude_sports: Optional[bool] = None,
    exclude_disasters: Optional[bool] = None,
    apply_langchain: bool = True
) -> pd.DataFrame:
    """
    Filters news DataFrame by sentiment score threshold, noise reduction (sports/disasters),
    and LangChain Gemini LLM market relevance filter (if gem_key is configured).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    min_score = min_score if min_score is not None else MIN_SENTIMENT_SCORE
    mode = (mode if mode is not None else SENTIMENT_MODE).lower()
    exclude_sports = exclude_sports if exclude_sports is not None else EXCLUDE_SPORTS
    exclude_disasters = exclude_disasters if exclude_disasters is not None else EXCLUDE_DISASTERS

    filtered_records = []
    for _, row in df.iterrows():
        # Score check
        score = float(row.get("sentiment_score", 0.0))
        if mode == "positive":
            if score < min_score:
                continue
        elif mode == "absolute":
            if abs(score) < min_score:
                continue
        else:
            if score < min_score:
                continue

        # Topic check (sports / disaster)
        headline = str(row.get("Headline", ""))
        desc = str(row.get("Description", ""))
        cat = str(row.get("Category", ""))
        full_text = f"{headline}. {desc}"

        blocked, _ = is_blacklisted_topic(
            full_text,
            category=cat,
            exclude_sports=exclude_sports,
            exclude_disasters=exclude_disasters
        )
        if not blocked:
            filtered_records.append(row)

    if not filtered_records:
        return pd.DataFrame(columns=df.columns)
    
    result_df = pd.DataFrame(filtered_records)
    
    # LangChain Gemini filtering if API key available
    if apply_langchain and not result_df.empty:
        api_key = get_gemini_api_key()
        if api_key:
            result_df = filter_news_with_langchain(result_df)
        else:
            print("ℹ️ Note: 'gem_key' not set in .env. Skipping LangChain Gemini LLM filtering (sentiment + noise filter applied).")

    return result_df


def load_alerts(csv_path=CSV_PATH, apply_filter=True):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Alerts CSV not found at '{csv_file}'.")
        return pd.DataFrame()
    df = pd.read_csv(csv_file)
    if apply_filter and not df.empty and "sentiment_score" in df.columns:
        df = filter_alerts(df, apply_langchain=False)
    return df


def format_alert_email_content(alerts_df):
    content = "🚨 The following news articles passed LangChain filtering and is likely to affect the stock markets:\n\n"

    for _, row in alerts_df.iterrows():
        headline = row.get("Relevant_Headline", row.get("Headline", "N/A"))
        score_tag = ""
        if "sentiment_score" in row and not pd.isna(row["sentiment_score"]):
            score = float(row["sentiment_score"])
            label = row.get("sentiment_label", "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral")
            score_tag = f"  [{label.upper()} | Polarity Score: {score:+.2f}]"
        elif "label" in row and not pd.isna(row["label"]):
            score_tag = f"  [{str(row['label']).upper()}]"

        content += f"- {headline}{score_tag}\n\n"
        explanation = row.get("llm_explanation") or row.get("explanation")
        if explanation and not pd.isna(explanation):
            content += f"  Explanation: {explanation}\n\n"
    return content


def send_email(alerts_df, dry_run=False):
    msg = EmailMessage()
    msg['Subject'] = f'🚨 Geo-Predict Alert — {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    content = format_alert_email_content(alerts_df)
    msg.set_content(content)

    if dry_run or not APP_PASSWORD:
        print("ℹ️ [DRY RUN / No APP_PASSWORD] Email alert prepared successfully:")
        print(f"Subject: {msg['Subject']}")
        print(f"To: {msg['To']}")
        print("Content preview:")
        print(content[:400] + ("..." if len(content) > 400 else ""))
        return True

    # Send email via SMTP
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email sent with alerts!")
        return True
    except Exception as e:
        print(f"⚠️ Failed to send email via SMTP: {e}")
        return False


def main(dry_run=False):
    alerts = load_alerts(CSV_PATH)
    if not alerts.empty:
        send_email(alerts, dry_run=dry_run)
    else:
        print("✅ No alerts to send.")


if __name__ == "__main__":
    main()
