
import os
import time
import tempfile
from datetime import datetime

import streamlit as st
from crewai import Crew, Process

from agents import (
    manager_agent,
    job_analyst_agent,
    cv_agent,
    research_agent,
    application_agent,
    interview_agent,
    critic_agent,
)

from tasks import TASKS
from tools import extract_cv_text
from memory import CareerMemory


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CareerOps AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .agent-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.75rem;
    }

    .status-box {
        padding: 1rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .result-box {
        padding: 1.25rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


AGENT_DESCRIPTIONS = {
    "Manager": (
        "Get a structured overview of your career situation, "
        "job requirements, strengths, gaps, and next steps."
    ),
    "Job Analyst": (
        "Analyze the job description, requirements, skills, "
        "keywords, responsibilities, and qualifications."
    ),
    "CV Analyst": (
        "Compare your CV against the target job and identify "
        "matches, gaps, and CV improvement opportunities."
    ),
    "Research Agent": (
        "Analyze company and opportunity information supplied "
        "in your job description and career request."
    ),
    "Application Agent": (
        "Create tailored application guidance, professional "
        "summary, CV improvements, and cover-letter content."
    ),
    "Interview Agent": (
        "Generate technical, behavioral, situational, and "
        "job-specific interview preparation."
    ),
    "Critic Agent": (
        "Review your application information and identify "
        "weaknesses, missing evidence, gaps, and improvements."
    ),
}


# ============================================================
# SESSION STATE
# ============================================================

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "career_request" not in st.session_state:
    st.session_state.career_request = ""

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Job Analyst"

if "result" not in st.session_state:
    st.session_state.result = ""

if "result_agent" not in st.session_state:
    st.session_state.result_agent = ""

if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None


# ============================================================
# RESET FUNCTIONS
# ============================================================

def reset_result():
    st.session_state.result = ""
    st.session_state.result_agent = ""
    st.session_state.last_run_time = None


def reset_everything():
    st.session_state.cv_text = ""
    st.session_state.job_description = ""
    st.session_state.career_request = ""
    st.session_state.selected_agent = "Job Analyst"
    st.session_state.result = ""
    st.session_state.result_agent = ""
    st.session_state.last_run_time = None


# ============================================================
# GROQ ERROR DETECTION
# ============================================================

def is_quota_error(error):
    """Detect Groq quota/rate-limit errors."""

    message = str(error).upper()

    quota_patterns = [
        "RATE_LIMIT",
        "RATE LIMIT",
        "TOO MANY REQUESTS",
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA EXCEEDED",
        "LIMIT REACHED",
    ]

    return any(
        pattern in message
        for pattern in quota_patterns
    )


def is_model_not_found_error(error):
    """Detect invalid/unavailable model errors."""

    message = str(error).upper()

    return (
        "MODEL_NOT_FOUND" in message
        or (
            "MODEL" in message
            and "NOT FOUND" in message
        )
        or "404" in message
    )


def is_retryable_ai_error(error):
    """Return True only for temporary AI service errors."""

    if is_quota_error(error):
        return False

    if is_model_not_found_error(error):
        return False

    message = str(error).upper()

    retryable_patterns = [
        "503",
        "SERVICE_UNAVAILABLE",
        "UNAVAILABLE",
        "500",
        "502",
        "504",
        "INTERNAL SERVER ERROR",
        "BAD GATEWAY",
        "GATEWAY TIMEOUT",
    ]

    return any(
        pattern in message
        for pattern in retryable_patterns
    )


# ============================================================
# CREW EXECUTION WITH RETRY
# ============================================================

def kickoff_with_retry(crew, inputs, max_attempts=4):

    delays = [5, 15, 30]

    for attempt in range(max_attempts):

        try:
            return crew.kickoff(inputs=inputs)

        except Exception as error:

            if is_quota_error(error):
                raise

            if is_model_not_found_error(error):
                raise

            if not is_retryable_ai_error(error):
                raise

            if attempt == max_attempts - 1:
                raise

            delay = delays[
                min(attempt, len(delays) - 1)
            ]

            st.warning(
                f"Temporary AI service error. "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    return None


# ============================================================
# CREATE SINGLE-AGENT CREW
# ============================================================

def create_single_agent_crew(agent_name):

    if agent_name not in AGENTS:
        raise ValueError(
            f"Unknown agent: {agent_name}"
        )

    if agent_name not in TASKS:
        raise ValueError(
            f"No task configured for: {agent_name}"
        )

    selected_agent = AGENTS[agent_name]
    selected_task = TASKS[agent_name]

    crew = Crew(
        agents=[selected_agent],
        tasks=[selected_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# ============================================================
# RUN SELECTED AGENT
# ============================================================

def run_selected_agent(
    agent_name,
    cv_text,
    job_description,
    career_request,
):

    crew = create_single_agent_crew(agent_name)

    inputs = {
        "cv_text": cv_text,
        "job_description": job_description,
        "career_request": career_request,
    }

    memory = None

    try:
        memory = CareerMemory()
    except Exception:
        memory = None

    if memory is not None:

        try:
            if hasattr(memory, "set_cv"):
                memory.set_cv(cv_text)
        except Exception:
            pass

        try:
            if hasattr(memory, "set_job_description"):
                memory.set_job_description(
                    job_description
                )
        except Exception:
            pass

        try:
            if hasattr(memory, "set_career_request"):
                memory.set_career_request(
                    career_request
                )
        except Exception:
            pass

    result = kickoff_with_retry(
        crew=crew,
        inputs=inputs,
    )

    return result


# ============================================================
# EXTRACT RESULT TEXT
# ============================================================

def extract_result_text(result):

    if result is None:
        return ""

    if hasattr(result, "raw"):

        raw = result.raw

        if raw is not None:
            return str(raw)

    return str(result)


# ============================================================
# GROQ ERROR DISPLAY
# ============================================================

def show_quota_error(error):

    st.error(
        "Groq API rate limit or quota has been reached."
    )

    st.markdown(
        """
        Groq has temporarily limited this application's
        API usage.

        You can:

        - Wait for the rate limit to reset
        - Check your Groq usage and limits
        - Reduce the number of AI calls
        - Use an appropriate Groq plan
        """
    )

    with st.expander("Technical details"):
        st.code(str(error))


def show_model_error(error):

    st.error(
        "The configured Groq model is unavailable."
    )

    st.markdown(
        """
        CareerOps is configured to use the model
        specified by `GROQ_MODEL`.

        Check your Streamlit secrets and make sure the
        model ID is valid for your Groq account.
        """
    )

    with st.expander("Technical details"):
        st.code(str(error))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ CareerOps AI")

    st.markdown("---")

    st.subheader("Agent")

    agent_names = list(AGENTS.keys())

    selected_agent = st.selectbox(
        "Choose one agent",
        options=agent_names,
        index=agent_names.index(
            st.session_state.selected_agent
        ),
    )

    st.session_state.selected_agent = selected_agent

    st.caption(
        AGENT_DESCRIPTIONS[selected_agent]
    )

    st.markdown("---")

    st.subheader("System Status")

    try:
        secret_key_exists = bool(
            st.secrets.get("GROQ_API_KEY")
        )
    except Exception:
        secret_key_exists = False

    environment_key_exists = bool(
        os.getenv("GROQ_API_KEY")
    )

    groq_key_exists = (
        secret_key_exists
        or environment_key_exists
    )

    if groq_key_exists:
        st.success("Groq API key detected")
    else:
        st.error("Groq API key not detected")

    try:
        configured_model = (
            st.secrets.get("GROQ_MODEL")
            or os.getenv("GROQ_MODEL")
            or "openai/gpt-oss-20b"
        )
    except Exception:
        configured_model = (
            os.getenv("GROQ_MODEL")
            or "openai/gpt-oss-20b"
        )

    st.caption(
        f"Model: `{configured_model}`"
    )

    st.info(
        "Only the selected agent is executed when you click Run."
    )

    st.markdown("---")

    if st.button(
        "🗑️ Reset Everything",
        use_container_width=True,
    ):
        reset_everything()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-header">🎯 CareerOps AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    "AI-powered career operations assistant"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# CURRENT AGENT DISPLAY
# ============================================================

st.markdown(
    f"""
    <div class="agent-card">
        <strong>Selected Agent:</strong> {selected_agent}<br>
        <span>{AGENT_DESCRIPTIONS[selected_agent]}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CV SECTION
# ============================================================

st.header("📄 Candidate CV")

cv_upload = st.file_uploader(
    "Upload your CV",
    type=[
        "pdf",
        "docx",
        "txt",
    ],
    help="Upload a PDF, DOCX, or TXT CV.",
)


if cv_upload is not None:

    try:

        file_extension = (
            os.path.splitext(cv_upload.name)[1]
            .lower()
        )

        if file_extension == ".txt":

            st.session_state.cv_text = (
                cv_upload.read()
                .decode(
                    "utf-8",
                    errors="ignore",
                )
            )

        else:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension,
            ) as temp_file:

                temp_file.write(
                    cv_upload.getbuffer()
                )

                temp_path = temp_file.name

            try:

                extracted_text = extract_cv_text(
                    temp_path
                )

                if extracted_text:
                    st.session_state.cv_text = (
                        extracted_text
                    )

            finally:

                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        st.success(
            f"CV loaded: {cv_upload.name}"
        )

    except Exception as error:

        st.error(
            f"Could not read the uploaded CV: {error}"
        )


cv_text_input = st.text_area(
    "Or paste your CV here",
    value=st.session_state.cv_text,
    height=300,
    placeholder="Paste your CV text here...",
)

st.session_state.cv_text = cv_text_input


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header("💼 Job Description")

job_description_input = st.text_area(
    "Paste the target job description",
    value=st.session_state.job_description,
    height=300,
    placeholder="Paste the complete job description here...",
)

st.session_state.job_description = job_description_input


# ============================================================
# CAREER REQUEST
# ============================================================

st.header("🎯 Career Request")

career_request_input = st.text_area(
    "What do you want CareerOps AI to help you with?",
    value=st.session_state.career_request,
    height=180,
    placeholder=(
        "Example:\n"
        "Analyze my fit for this position and tell me "
        "what I should improve before applying."
    ),
)

st.session_state.career_request = career_request_input


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs():

    errors = []

    if not st.session_state.cv_text.strip():
        errors.append(
            "Please provide your CV."
        )

    if not st.session_state.job_description.strip():
        errors.append(
            "Please provide the job description."
        )

    if not st.session_state.career_request.strip():
        errors.append(
            "Please describe what you want the agent to do."
        )

    return errors


# ============================================================
# RUN BUTTON
# ============================================================

st.markdown("---")

run_button = st.button(
    f"🚀 Run {selected_agent}",
    type="primary",
    use_container_width=True,
)


if run_button:

    validation_errors = validate_inputs()

    if validation_errors:

        for error in validation_errors:
            st.warning(error)

    else:

        reset_result()

        st.session_state.selected_agent = selected_agent

        with st.spinner(
            f"Running {selected_agent}..."
        ):

            try:

                result = run_selected_agent(
                    agent_name=selected_agent,
                    cv_text=st.session_state.cv_text,
                    job_description=(
                        st.session_state.job_description
                    ),
                    career_request=(
                        st.session_state.career_request
                    ),
                )

                result_text = extract_result_text(
                    result
                )

                st.session_state.result = result_text

                st.session_state.result_agent = (
                    selected_agent
                )

                st.session_state.last_run_time = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                st.success(
                    f"{selected_agent} completed successfully."
                )

            except Exception as error:

                if is_model_not_found_error(error):

                    show_model_error(error)

                elif is_quota_error(error):

                    show_quota_error(error)

                else:

                    st.error(
                        "The selected agent could not complete "
                        "the request."
                    )

                    with st.expander(
                        "Technical error details"
                    ):
                        st.code(str(error))


# ============================================================
# RESULTS
# ============================================================

if st.session_state.result:

    st.markdown("---")

    st.header(
        f"📊 {st.session_state.result_agent} Result"
    )

    if st.session_state.last_run_time:

        st.caption(
            f"Completed: "
            f"{st.session_state.last_run_time}"
        )

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        st.session_state.result
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    safe_agent_name = (
        st.session_state.result_agent
        .lower()
        .replace(" ", "_")
    )

    file_name = (
        f"careerops_{safe_agent_name}_report.txt"
    )

    st.download_button(
        label="⬇️ Download Result",
        data=st.session_state.result,
        file_name=file_name,
        mime="text/plain",
        use_container_width=True,
    )

