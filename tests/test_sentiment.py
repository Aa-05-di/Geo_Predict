"""
Unit and Integration Tests for GeoPredict Sentiment Analysis Pipeline.
Validates RoBERTa inference, polarity scoring in [-1.0, 1.0], DataFrame enrichment,
and alert formatting.
"""

from pathlib import Path
import pytest
import pandas as pd
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.roberta_sentiment import RobertaSentimentAnalyzer, analyze_sentiment
from models.model import (
    load_alerts,
    format_alert_email_content,
    CSV_PATH,
    filter_alerts,
    is_blacklisted_topic
)


@pytest.fixture(scope="module")
def analyzer():
    """Initializes a shared analyzer instance for tests."""
    return RobertaSentimentAnalyzer()


def test_positive_headline_sentiment(analyzer):
    text = "Company reports record profits, surging revenue, and raises annual guidance."
    res = analyzer.predict(text)

    assert "sentiment_score" in res
    assert "label" in res
    assert "confidence" in res
    assert -1.0 <= res["sentiment_score"] <= 1.0
    assert res["label"] == "Positive"
    assert res["sentiment_score"] > 0.0


def test_negative_headline_sentiment(analyzer):
    text = "Severe financial distress as firm files for bankruptcy amid cascading debt crisis."
    res = analyzer.predict(text)

    assert "sentiment_score" in res
    assert "label" in res
    assert -1.0 <= res["sentiment_score"] <= 1.0
    assert res["label"] == "Negative"
    assert res["sentiment_score"] < 0.0


def test_neutral_headline_sentiment(analyzer):
    text = "The central committee will hold its scheduled meeting on Tuesday afternoon."
    res = analyzer.predict(text)

    assert -1.0 <= res["sentiment_score"] <= 1.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == 3


def test_batch_prediction_consistency(analyzer):
    headlines = [
        "Tech giant posts blockbuster earnings beats across all segments.",
        "Retail sales plummet sharply as economic downturn deepens.",
        "Treasury department to announce debt auction schedule tomorrow."
    ]
    results = analyzer.predict_batch(headlines)

    assert len(results) == len(headlines)
    for res in results:
        assert -1.0 <= res["sentiment_score"] <= 1.0
        assert res["label"] in ["Positive", "Negative", "Neutral"]
        assert 0.0 <= res["confidence"] <= 1.0


def test_analyze_sentiment_backward_compatibility(analyzer):
    headlines = ["Shares soar after successful takeover bid.", "Factory closes laying off hundreds."]
    res = analyze_sentiment(headlines, analyzer=analyzer)

    assert len(res) == 2
    assert "news" in res[0]
    assert "sentiment_score" in res[0]
    assert "label" in res[0]
    assert res[0]["sentiment_score"] > 0
    assert res[1]["sentiment_score"] < 0


def test_dataframe_enrichment(analyzer):
    df = pd.DataFrame({
        "Headline": [
            "Economy expands at fastest pace in two decades.",
            "Inflation hits 40-year high triggering emergency rate increases."
        ]
    })
    enriched = analyzer.analyze_dataframe(df, text_column="Headline")

    assert "sentiment_label" in enriched.columns
    assert "sentiment_score" in enriched.columns
    assert "sentiment_confidence" in enriched.columns
    assert len(enriched) == 2
    assert enriched.iloc[0]["sentiment_label"] == "Positive"
    assert enriched.iloc[1]["sentiment_label"] == "Negative"


def test_alert_formatting_with_sentiment():
    sample_alerts = pd.DataFrame({
        "Relevant_Headline": [
            "US Fed slashes interest rates boosting equity markets",
            "Crude oil tumbles on unexpected inventory build"
        ],
        "sentiment_score": [0.82, -0.65],
        "sentiment_label": ["Positive", "Negative"]
    })
    content = format_alert_email_content(sample_alerts)

    assert "US Fed slashes interest rates" in content
    assert "POSITIVE" in content
    assert "+0.82" in content
    assert "NEGATIVE" in content
    assert "-0.65" in content


def test_topic_blacklisting():
    assert is_blacklisted_topic("Mourinho leads team to victory", "European Football")[0] is True
    assert is_blacklisted_topic("South Africa test match cricket update", "Sport")[0] is True
    assert is_blacklisted_topic("Himalayan disasters put thousands at risk", "Asia")[0] is True
    assert is_blacklisted_topic("Deadly crash kills 14 in train accident", "Europe")[0] is True
    assert is_blacklisted_topic("Microsoft reports record profits and dividend hike", "Business")[0] is False


def test_filter_alerts_threshold_and_noise_reduction():
    test_df = pd.DataFrame([
        {"Headline": "Microsoft profits soar", "Description": "Tech giant beats earnings", "Category": "Business", "sentiment_score": 0.95},
        {"Headline": "Himalayan disasters hit villages", "Description": "Heavy rain caused landslide disaster", "Category": "Asia", "sentiment_score": 0.96},
        {"Headline": "Premier League match excitement", "Description": "Football team wins", "Category": "Football", "sentiment_score": 0.88},
        {"Headline": "Moderate economic growth reported", "Description": "GDP slightly up", "Category": "Economy", "sentiment_score": 0.35},
        {"Headline": "Market crash triggers massive selloff", "Description": "Stocks plunge sharply", "Category": "Markets", "sentiment_score": -0.85},
    ])

    # Mode: positive with threshold 0.70
    filtered = filter_alerts(test_df, min_score=0.70, mode="positive", exclude_sports=True, exclude_disasters=True)
    assert len(filtered) == 1
    assert filtered.iloc[0]["Headline"] == "Microsoft profits soar"

    # Verify score < 0.70 dropped
    assert "Moderate economic growth reported" not in filtered["Headline"].values
    # Verify disaster dropped despite high score (0.96)
    assert "Himalayan disasters hit villages" not in filtered["Headline"].values
    # Verify sports dropped despite high score (0.88)
    assert "Premier League match excitement" not in filtered["Headline"].values
    # Verify negative dropped under positive mode
    assert "Market crash triggers massive selloff" not in filtered["Headline"].values


if __name__ == "__main__":
    pytest.main(["-v", __file__])
