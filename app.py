import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Career & Learning Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Career & Learning Agent")
st.write("Your intelligent AI-powered career and learning assistant.")

# -----------------------------
# API Key
# -----------------------------

api_key = st.text_input(
    "🔑 Enter your OpenAI API Key",
    type="password"
)

if not api_key:
    st.info("Enter your API key to start the AI Agent.")
    st.stop()

# -----------------------------
# Agent State
# -----------------------------

class AgentState(TypedDict):
    user_input: str
    response: str


# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=api_key
)


# -----------------------------
# Agent Node
# -----------------------------

def career_agent(state: AgentState):

    prompt = f"""
You are an AI Career and Learning Agent.

Your job is to help the user create a practical
learning and career plan.

User request:
{state['user_input']}

Analyze the request and provide:

1. Goal identification
2. Required skills
3. Learning roadmap
4. Recommended projects
5. Practical tasks
6. Suggested timeline
7. Next action

Keep the response practical and beginner-friendly.
"""

    messages = [
        SystemMessage(
            content="You are a helpful AI Career Agent."
        ),
        HumanMessage(content=prompt)
    ]

    result = llm.invoke(messages)

    return {
        "response": result.content
    }


# -----------------------------
# Build Agent Graph
# -----------------------------

graph = StateGraph(AgentState)

graph.add_node("career_agent", career_agent)

graph.add_edge(START, "career_agent")
graph.add_edge("career_agent", END)

agent = graph.compile()


# -----------------------------
# User Interface
# -----------------------------

st.subheader("🎯 Tell the Agent Your Goal")

user_input = st.text_area(
    "Example: I want to become an AI Engineer. Create a 6-month roadmap for me.",
    height=150
)

if st.button("🚀 Run AI Agent"):

    if not user_input.strip():
        st.warning("Please enter your goal.")
        st.stop()

    with st.spinner("🤖 Agent is thinking..."):

        result = agent.invoke({
            "user_input": user_input,
            "response": ""
        })

    st.success("Agent completed the task!")

    st.subheader("📋 Agent Response")

    st.markdown(result["response"])