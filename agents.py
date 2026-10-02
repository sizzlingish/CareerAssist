import os
import streamlit as st
from crewai import Agent, LLM


def get_groq_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY")
    if key:
        return key.strip()

    return None


GROQ_API_KEY = get_groq_api_key()

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured.\n\n"
        "For Streamlit Cloud, add GROQ_API_KEY "
        "under App Settings → Secrets."
    )


# Read optional model configuration
try:
    secret_model = st.secrets.get("GROQ_MODEL")
except Exception:
    secret_model = None

raw_model = (
    os.getenv("GROQ_MODEL")
    or secret_model
    or "openai/gpt-oss-20b"
)

# Fix the old/short model ID automatically
if raw_model.strip() == "gpt-oss-20b":
    raw_model = "openai/gpt-oss-20b"

GROQ_MODEL = raw_model.strip()


groq_llm = LLM(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


COMMON_AGENT_SETTINGS = {
    "llm": groq_llm,
    "verbose": True,
}
