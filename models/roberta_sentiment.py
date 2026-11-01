"""
RoBERTa Sentiment Analysis Module for GeoPredict.
Replaces legacy NLTK VADER with a transformer-based sequence classification model
specialized in financial and economic market sentiment.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd


# Default path where the fine-tuned model checkpoint is stored
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "roberta_finetuned"
FALLBACK_PRETRAINED_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DEFAULT_ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
DEFAULT_LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}


class RobertaSentimentAnalyzer:
    """
    RoBERTa-based sentiment analyzer for geopolitical and financial news.
    Computes discrete sentiment categories (Positive, Neutral, Negative)
    and continuous compound polarity scores in [-1.0, 1.0].
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_length: int = 128
    ):
        self.max_length = max_length
        self.device = self._select_device(device)
        self.model_path = self._resolve_model_path(model_path)

        print(f"Loading RoBERTa Sentiment Model from '{self.model_path}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        # Extract or harmonize id2label mapping
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            raw_id2label = self.model.config.id2label
            self.id2label = {}
            for k, v in raw_id2label.items():
                idx = int(k)
                v_lower = str(v).lower()
                if "pos" in v_lower:
                    self.id2label[idx] = "Positive"
                elif "neg" in v_lower:
                    self.id2label[idx] = "Negative"
                else:
                    self.id2label[idx] = "Neutral"
        else:
            self.id2label = DEFAULT_ID2LABEL

        self.label2id = {v: k for k, v in self.id2label.items()}

    @staticmethod
    def _select_device(requested_device: Optional[str] = None) -> torch.device:
        if requested_device:
            return torch.device(requested_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _resolve_model_path(model_path: Optional[Union[str, Path]]) -> Union[str, Path]:
        if model_path:
            path = Path(model_path)
            if path.exists():
                return path
            return model_path  # Might be a HuggingFace hub model ID

        if DEFAULT_MODEL_DIR.exists() and (DEFAULT_MODEL_DIR / "config.json").exists():
            return DEFAULT_MODEL_DIR

        # Fallback to high-quality pretrained sentiment transformer if fine-tuned checkpoint not yet trained
        return FALLBACK_PRETRAINED_MODEL

    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text string.
        Returns label, confidence, continuous score in [-1.0, 1.0], and class probabilities.
        """
        results = self.predict_batch([text], batch_size=1)
        return results[0]

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Run batched inference on a list of text strings.
        """
        if not texts:
            return []

        cleaned_texts = [str(t) if (t is not None and not pd.isna(t)) else "" for t in texts]
        results = []

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            for item_idx, prob in enumerate(probs):
                pred_id = int(prob.argmax())
                pred_label = self.id2label.get(pred_id, "Neutral")
                confidence = float(prob[pred_id])

                # Resolve positive and negative probabilities
                pos_idx = self.label2id.get("Positive", 2)
                neg_idx = self.label2id.get("Negative", 0)
                pos_prob = float(prob[pos_idx]) if pos_idx < len(prob) else 0.0
                neg_prob = float(prob[neg_idx]) if neg_idx < len(prob) else 0.0
                neu_idx = self.label2id.get("Neutral", 1)
                neu_prob = float(prob[neu_idx]) if neu_idx < len(prob) else 0.0

                # Continuous compound score in [-1.0, 1.0]
                compound_score = round(pos_prob - neg_prob, 4)

                results.append({
                    "label": pred_label,
                    "sentiment_score": compound_score,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "negative": round(neg_prob, 4),
                        "neutral": round(neu_prob, 4),
                        "positive": round(pos_prob, 4)
                    }
                })

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_prefix: str = "sentiment_"
    ) -> pd.DataFrame:
        """
        Add sentiment predictions directly to a Pandas DataFrame.
        """
        df_copy = df.copy()
        texts = df_copy[text_column].tolist()
        predictions = self.predict_batch(texts)

        df_copy[f"{output_prefix}label"] = [p["label"] for p in predictions]
        df_copy[f"{output_prefix}score"] = [p["sentiment_score"] for p in predictions]
        df_copy[f"{output_prefix}confidence"] = [p["confidence"] for p in predictions]
        return df_copy


_GLOBAL_ANALYZER: Optional[RobertaSentimentAnalyzer] = None

def get_analyzer(model_path: Optional[Union[str, Path]] = None) -> RobertaSentimentAnalyzer:
    """Singleton getter for RobertaSentimentAnalyzer."""
    global _GLOBAL_ANALYZER
    if _GLOBAL_ANALYZER is None:
        _GLOBAL_ANALYZER = RobertaSentimentAnalyzer(model_path=model_path)
    return _GLOBAL_ANALYZER


def analyze_sentiment(news_list: List[str], analyzer: Optional[RobertaSentimentAnalyzer] = None) -> List[Dict]:
    """
    Drop-in replacement for the legacy NLTK VADER analyze_sentiment function.
    Given a list of news headlines/articles, returns a list of dictionaries with:
    {
        'news': news_text,
        'sentiment_score': compound_score,  # in [-1.0, 1.0]
        'label': 'Positive' | 'Negative' | 'Neutral',
        'confidence': float
    }
    """
    if analyzer is None:
        analyzer = get_analyzer()

    predictions = analyzer.predict_batch(news_list)
    results = []
    for news, pred in zip(news_list, predictions):
        results.append({
            "news": news,
            "sentiment_score": pred["sentiment_score"],
            "label": pred["label"],
            "confidence": pred["confidence"]
        })
    return results
