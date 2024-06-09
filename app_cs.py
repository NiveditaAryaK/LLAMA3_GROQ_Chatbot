import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompt = {
    "role": "system",
    "content": "You are a helpful assistant. You reply with efficient answers. Ask for additional information if you are not confident about the answer.",
}

def chat_groq(message, summary):
    messages = [system_prompt]
    if summary:
        messages.append({"role": "system", "content": f"Summary of previous conversation: {summary}"})
    messages.append({"role": "user", "content": message})

    # Print the current output prompt
    print("Current Output Prompt:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")

    response_content = ""
    stream = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content
    return response_content

def summarize_chat(history):
    summary = ""
    if history:
        all_messages = []
        for role, content in history:
            all_messages.append(f"{role}: {content}")
        summary_prompt = "Summarize the following conversation:\n" + "\n".join(all_messages)
        response_content = chat_groq(summary_prompt, "")  # Pass empty summary for summary prompt
        summary = response_content.strip()  
    return summary

# Streamlit app setup
st.title("ChatBot")

if "history" not in st.session_state:
    st.session_state["history"] = []

if "summary" not in st.session_state:
    st.session_state["summary"] = ""

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

def submit():
    user_message = st.session_state["user_input"]
    history = st.session_state["history"]
    summary = st.session_state["summary"]
    if user_message:
        current_prompt = chat_groq(user_message, history)
        history.append((user_message, current_prompt))
        # st.session_state["history"] = history[:10]  #History limited to last 10 entries
        st.session_state["user_input"] = ""  # Clear the input field
        st.session_state["current_prompt"] = current_prompt
        st.session_state["summary"] = summarize_chat(st.session_state["history"])

# Input text field
st.text_area("Enter your message:", key="user_input", on_change=submit)

# Display the current output prompt if available
if "current_prompt" in st.session_state:
    st.write(st.session_state["current_prompt"])

# Display Chat Summary with expander
with st.expander("Chat Summary"):
    st.write(st.session_state["summary"])

# Display Chat History with expander
with st.expander("Chat History (Last 4)"):
    for i, (user_msg, assistant_msg) in enumerate(st.session_state["history"][-4:]):
        st.write(f"{i+1}. User: {user_msg}")
        st.write(f"  Assistant: {assistant_msg}")
