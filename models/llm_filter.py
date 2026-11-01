"""
LangChain Google GenAI (Gemini) filtering module for GeoPredict.
Restores the original prompt and LLM relevance filtering:
Prompt: "Is the following news article likely to influence stock markets?If yes,state yes and explain how.\n\nArticle:{news_text}"
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "notebooks" / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

_LLM_INSTANCE = None


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment (.env)."""
    return os.getenv("gem_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def init_langchain_gemini(model: Optional[str] = None, temperature: float = 0.6):
    """
    Initializes the ChatGoogleGenerativeAI instance using the API key in .env.
    Defaults to gemini-2.5-flash-lite with fast response and robust timeouts.
    """
    global _LLM_INSTANCE
    api_key = get_gemini_api_key()
    if not api_key:
        print("⚠️ [LangChain] Neither 'gem_key' nor 'GOOGLE_API_KEY' found in notebooks/.env.")
        print("   Please set `gem_key=your_gemini_api_key` in notebooks/.env to enable LLM market filtering.")
        return None

    os.environ["GOOGLE_API_KEY"] = api_key
    candidate_models = [model] if model else [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
    ]

    from langchain_google_genai import ChatGoogleGenerativeAI

    for m in candidate_models:
        if not m:
            continue
        try:
            _LLM_INSTANCE = ChatGoogleGenerativeAI(
                model=m,
                temperature=temperature,
                max_retries=1,
                timeout=60
            )
            return _LLM_INSTANCE
        except Exception:
            continue

    print("⚠️ [LangChain] Failed to initialize ChatGoogleGenerativeAI with candidate models.")
    return None


def get_llm_relevance(news_text: str, llm=None) -> Tuple[bool, str]:
    """
    Queries Gemini using the restored original prompt:
    'Is the following news article likely to influence stock markets?If yes,state yes and explain how.\n\nArticle:{news_text}'
    Returns (is_relevant: bool, explanation: str).
    """
    if llm is None:
        global _LLM_INSTANCE
        if _LLM_INSTANCE is None:
            _LLM_INSTANCE = init_langchain_gemini()
        llm = _LLM_INSTANCE

    if llm is None:
        # LLM not initialized due to missing API key
        return False, "Gemini API key not configured."

    prompt = f"Is the following news article likely to influence stock markets?If yes,state yes and explain how. Dont use too much words. give exact amount of information in minimal words\n\nArticle:{news_text}"
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        is_relevant = "yes" in content.lower()
        return is_relevant, content
    except Exception as e:
        print(f"⚠️ [LangChain] Error querying Gemini: {e}")
        return False, str(e)


def filter_news_with_langchain(news_df: pd.DataFrame, text_column: str = "Headline", desc_column: str = "Description") -> pd.DataFrame:
    """
    Applies the restored LangChain Gemini market relevance filter to a DataFrame.
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame()

    llm = init_langchain_gemini()
    if llm is None:
        print("ℹ️ Skipping LangChain Gemini filtering: 'gem_key' not set in notebooks/.env.", flush=True)
        return news_df

    relevant_rows = []
    print(f"Evaluating {len(news_df)} articles with LangChain Gemini ({llm.model})...", flush=True)
    for idx, (_, row) in enumerate(news_df.iterrows(), 1):
        text = str(row.get(text_column, ""))
        desc = str(row.get(desc_column, ""))
        combined = f"{text}. {desc}" if desc and desc != "nan" else text

        print(f"  [{idx}/{len(news_df)}] Evaluating: '{text[:50]}...' ", end="", flush=True)
        try:
            is_rel, explanation = get_llm_relevance(combined, llm=llm)
            if "RESOURCE_EXHAUSTED" in explanation or "quota" in explanation.lower():
                print(f"\n⚠️ [LangChain] API quota reached. Passing candidate articles through.", flush=True)
                return news_df

            status_str = "RELEVANT" if is_rel else "NOT RELEVANT"
            print(f"-> {status_str}", flush=True)

            if is_rel:
                row_dict = row.to_dict()
                row_dict["llm_explanation"] = explanation
                relevant_rows.append(row_dict)
        except Exception as e:
            print(f"-> EXCEPTION ({e})", flush=True)
            row_dict = row.to_dict()
            relevant_rows.append(row_dict)

    if not relevant_rows:
        return pd.DataFrame(columns=list(news_df.columns) + ["llm_explanation"])
    return pd.DataFrame(relevant_rows)
