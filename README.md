# CareerOps AI – AI Career Assistant

CareerOps AI is a multi-agent AI career assistant that helps users analyze job opportunities, improve their CV, research companies, create tailored application materials, and prepare for interviews.

The project uses a coordinated team of specialized AI agents powered by Google Gemini and CrewAI, with Streamlit providing the user interface.

## 🚀 Features

- 📄 CV analysis
- 💼 Job description analysis
- 🔍 Required and preferred skill extraction
- 🏢 Company and opportunity research
- 🎯 CV-to-job matching
- ✍️ Tailored professional summary and application content
- 📝 Cover letter generation
- 🎤 Technical and behavioral interview preparation
- 💡 Suggested questions to ask the interviewer
- 🔎 Critic/reviewer agent for quality checking
- 🔄 Feedback and improvement workflow
- 🧠 Career workflow memory

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      Streamlit      │
                    │     User Interface  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Manager Agent    │
                    │ Workflow Coordinator│
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │  Job Analyst   │   │   CV Analyst    │   │ Research Agent │
 └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Application Agent   │
                    │ CV + Cover Letter   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Interview Agent     │
                    │ Preparation         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Critic Agent     │
                    │ Quality & Gap Check │
                    └──────────┬──────────┘
                               │
                       Feedback / Revision
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Final Results     │
                    └─────────────────────┘
