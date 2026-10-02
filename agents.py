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
    raise ValueError("GROQ_API_KEY is not configured.")


# Fixed Groq model
GROQ_MODEL = "openai/gpt-oss-20b"

groq_llm = LLM(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


COMMON_AGENT_SETTINGS = {
    "llm": groq_llm,
    "verbose": True,
}
