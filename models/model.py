import pandas as pd
import smtplib
from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv
import os
from pathlib import Path
env_path = Path("/Users/aadi/GEO_PREDICT/Geo_Predict/notebooks/.env")
load_dotenv(dotenv_path=env_path)

CSV_PATH = "/Users/aadi/GEO_PREDICT/Geo_Predict/notebooks/rsentiments.csv"
SENDER_EMAIL = "aadithyanlearn@gmail.com"
RECEIVER_EMAIL = "maitharockers@gmail.com"
APP_PASSWORD = os.getenv("gmail")

def load_alerts(csv_path):
    df = pd.read_csv(csv_path)
    return df

def send_email(alerts_df):
    msg = EmailMessage()
    msg['Subject'] = f'🚨 Geo-Predict Alert — {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    content = "🚨 The following news articles passed LangChain filtering and is likely to affect the stock markets:\n\n"
    for _, row in alerts_df.iterrows():
        content += f"- {row['Relevant_Headline']}\n\n"

    msg.set_content(content)

    # Send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)

    print("✅ Email sent with alerts!")

def main():
    alerts = load_alerts(CSV_PATH)
    if not alerts.empty:
        send_email(alerts)
    else:
        print("✅ No alerts to send.")

if __name__ == "__main__":
    main()
