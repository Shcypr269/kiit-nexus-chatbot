import streamlit as st
from rag_chain import build_chain

# Page config
st.set_page_config(
    page_title="KIITBot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 KIITBot — Local Test")
st.caption("Powered by KIIT NEXUS · Free Stack: Groq + ChromaDB + HuggingFace")
st.divider()

# Build chain once and cache it in session
if "chain" not in st.session_state:
    with st.spinner("Loading KIITBot... (first load takes ~30 seconds)"):
        st.session_state.chain = build_chain()
        st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"📄 Sources: {', '.join(msg['sources'])}")

# Input
user_input = st.chat_input("Ask anything about KIIT...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result  = st.session_state.chain({"question": user_input})
            answer  = result["answer"]
            sources = list({
                d.metadata.get("source", "unknown")
                for d in result.get("source_documents", [])
            })

        st.write(answer)
        if sources:
            st.caption(f"📄 Sources: {', '.join(sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })