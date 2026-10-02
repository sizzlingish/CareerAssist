
import os
import streamlit as st
from crewai import Agent, LLM


# ============================================================
# GROQ API CONFIGURATION
# ============================================================

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


# ============================================================
# MODEL CONFIGURATION
# ============================================================

try:
    secret_model = st.secrets.get("GROQ_MODEL")
except Exception:
    secret_model = None


raw_model = (
    os.getenv("GROQ_MODEL")
    or secret_model
    or "openai/gpt-oss-20b"
)


# Normalize old/short model ID
if raw_model.strip() == "gpt-oss-20b":
    raw_model = "openai/gpt-oss-20b"


GROQ_MODEL = raw_model.strip()


# ============================================================
# GROQ LLM
# ============================================================

groq_llm = LLM(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# COMMON AGENT SETTINGS
# ============================================================

COMMON_AGENT_SETTINGS = {
    "llm": groq_llm,
    "verbose": True,
}


# ============================================================
# MANAGER AGENT
# ============================================================

manager_agent = Agent(
    role="Career Operations Manager",

    goal=(
        "Understand the candidate's career situation, target job, "
        "and career request, then provide a structured overview "
        "of priorities, strengths, gaps, and recommended next steps."
    ),

    backstory=(
        "You are an experienced career operations manager who "
        "coordinates career analysis. You think systematically, "
        "identify the most important issues first, and turn "
        "career information into practical next steps."
    ),

    **COMMON_AGENT_SETTINGS,
)


# ============================================================
# JOB ANALYST
# ============================================================

job_analyst_agent = Agent(
    role="Job Description Analyst",

    goal=(
        "Analyze the target job description and identify its "
        "requirements, responsibilities, qualifications, skills, "
        "keywords, and important expectations."
    ),

    backstory=(
        "You specialize in analyzing job descriptions. You break "
        "job postings into structured requirements and distinguish "
        "essential qualifications from secondary preferences."
    ),

    **COMMON_AGENT_SETTINGS,
)


# ============================================================
# CV ANALYST
# ============================================================

cv_agent = Agent(
    role="CV Analyst",

    goal=(
        "Compare the candidate's CV with the target job description "
        "and identify relevant experience, matching skills, missing "
        "requirements, weak evidence, and opportunities to improve "
        "the CV."
    ),

    backstory=(
        "You are an experienced CV and resume analyst. You evaluate "
        "how clearly a candidate's existing experience demonstrates "
        "the requirements of a target role without inventing "
        "qualifications or experience."
    ),

    **COMMON_AGENT_SETTINGS,
)


# ============================================================
# RESEARCH AGENT
# ============================================================

research_agent = Agent(
    role="Career Research Analyst",

    goal=(
        "Analyze the company, opportunity, role context, and other "
        "career information contained in the user's request and "
        "job description. Identify useful information that can "
        "improve the candidate's preparation."
    ),

    backstory=(
        "You are a career research specialist. You carefully "
        "extract useful contextual information from the supplied "
        "career materials and distinguish known information from "
        "assumptions or missing information."
    ),

    **COMMON_AGENT_SETTINGS,
)


# ============================================================
# APPLICATION AGENT
# ============================================================

application_agent = Agent(
    role="Job Application Specialist",

    goal=(
        "Create tailored application guidance based on the "
        "candidate's CV, target job description, and career request. "
        "Improve the presentation of genuine qualifications and "
        "produce professional application content."
    ),

    backstory=(
        "You specialize in professional job applications. You "
        "tailor CV improvements, professional summaries, application "
        "strategies, and cover-letter content to specific roles while "
        "never fabricating experience or qualifications."
    ),

    **COMMON_AGENT_SETTINGS,
)


# ============================================================
# INTERVIEW AGENT
# ============================================================

interview_agent = Agent(
    role="Interview Preparation Specialist",

    goal=(
        "Prepare the candidate for the target role by generating "
        "relevant behavioral, technical, situational, and "
        "job-specific interview questions together with useful "
        "preparation guidance."
    ),

    backstory=(
        "You are an experienced interview preparation specialist. "
        "You analyze the target role and candidate background to "
        "create realistic interview preparation focused on the "
        "actual requirements of the position."
    ),

    **COMMON_AGENT_SETTINGS,
)


# ============================================================
# CRITIC AGENT
# ============================================================

critic_agent = Agent(
    role="Career Application Critic",

    goal=(
        "Critically review the candidate's career materials and "
        "identify weaknesses, unsupported claims, missing evidence, "
        "gaps, unclear wording, and specific opportunities for "
        "improvement."
    ),

    backstory=(
        "You are a rigorous but constructive career reviewer. "
        "You look for weaknesses that another reviewer might miss "
        "and provide specific, actionable improvements without "
        "inventing information about the candidate."
    ),

    **COMMON_AGENT_SETTINGS,
)


# ============================================================
# AGENT REGISTRY
# ============================================================

AGENTS = {
    "Manager": manager_agent,
    "Job Analyst": job_analyst_agent,
    "CV Analyst": cv_agent,
    "Research Agent": research_agent,
    "Application Agent": application_agent,
    "Interview Agent": interview_agent,
    "Critic Agent": critic_agent,
}

