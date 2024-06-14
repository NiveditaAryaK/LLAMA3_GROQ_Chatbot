from llama3Model import chat_groq,summarize_chat
import streamlit as st
# Streamlit app setup
st.title("Hindi To English Translator")

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
st.text_area("Enter your text:", key="user_input", on_change=submit)

# Display the current output prompt if available
if "current_prompt" in st.session_state:
    st.write(st.session_state["current_prompt"])

# Display Chat Summary with expander
with st.expander("translation"):
    st.write(st.session_state["summary"])

# Display Chat History with expander
with st.expander("Chat History (Last 4)"):
    for i, (user_msg, assistant_msg) in enumerate(st.session_state["history"][-4:]):
        st.write(f"{i+1}. User: {user_msg}")
        st.write(f"  Assistant: {assistant_msg}")
