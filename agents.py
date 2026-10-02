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


GROQ_MODEL = (
    os.getenv("GROQ_MODEL")
    or st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b")
)


groq_llm = LLM(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


COMMON_AGENT_SETTINGS = {
    "llm": groq_llm,
    "verbose": True,
}


manager_agent = Agent(
    role="Career Operations Manager",
    goal=(
        "Understand the candidate's career situation, target role, "
        "and requested outcome, then provide a structured career plan."
    ),
    backstory=(
        "You are an experienced career operations manager who coordinates "
        "career analysis and application strategy."
    ),
    **COMMON_AGENT_SETTINGS,
)


job_analyst_agent = Agent(
    role="Job Description Analyst",
    goal=(
        "Analyze the target job description and identify requirements, "
        "skills, keywords, responsibilities, and qualifications."
    ),
    backstory=(
        "You specialize in extracting hiring requirements and translating "
        "job descriptions into actionable candidate requirements."
    ),
    **COMMON_AGENT_SETTINGS,
)


cv_agent = Agent(
    role="CV Analyst",
    goal=(
        "Compare the candidate's CV with the target job and identify "
        "matches, gaps, missing evidence, and improvement opportunities."
    ),
    backstory=(
        "You are an expert CV reviewer focused on evidence-based alignment "
        "between candidate experience and job requirements."
    ),
    **COMMON_AGENT_SETTINGS,
)


research_agent = Agent(
    role="Career Research Agent",
    goal=(
        "Analyze company and opportunity information available in the "
        "candidate's career request and job description."
    ),
    backstory=(
        "You are a career research specialist who extracts useful "
        "context from supplied opportunity information."
    ),
    **COMMON_AGENT_SETTINGS,
)


application_agent = Agent(
    role="Job Application Specialist",
    goal=(
        "Create tailored application guidance, professional summaries, "
        "CV improvements, and cover-letter content."
    ),
    backstory=(
        "You specialize in tailoring professional application materials "
        "to specific job opportunities."
    ),
    **COMMON_AGENT_SETTINGS,
)


interview_agent = Agent(
    role="Interview Preparation Specialist",
    goal=(
        "Prepare the candidate for technical, behavioral, situational, "
        "and job-specific interview questions."
    ),
    backstory=(
        "You are an experienced interview coach who creates realistic "
        "and role-specific interview preparation."
    ),
    **COMMON_AGENT_SETTINGS,
)


critic_agent = Agent(
    role="Career Application Critic",
    goal=(
        "Critically review the candidate's application information and "
        "identify weaknesses, missing evidence, gaps, and improvements."
    ),
    backstory=(
        "You are a rigorous reviewer who looks for weaknesses and "
        "actionable ways to strengthen career applications."
    ),
    **COMMON_AGENT_SETTINGS,
)
