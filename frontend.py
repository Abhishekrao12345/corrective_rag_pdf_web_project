
"""
frontend.py - Streamlit website only. Talks to backend.py over HTTP.
Run with:  streamlit run frontend.py
"""
import requests
import streamlit as st

API = "http://localhost:8002"

st.set_page_config(page_title="Corrective RAG: PDF + Web", page_icon="📄")
st.title("📄 Corrective RAG: PDF + Web")

# ---- sidebar: upload ----
with st.sidebar:
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf and st.session_state.get("pdf_name") != pdf.name:
        with st.spinner("Indexing PDF..."):
            try:
                r = requests.post(
                    f"{API}/upload",
                    files={"file": (pdf.name, pdf.getvalue(), "application/pdf")},
                    timeout=300,
                )
                r.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend. Is it running? ({e})")
                st.stop()
            st.session_state.session_id = r.json()["session_id"]
            st.session_state.pdf_name = pdf.name
            st.session_state.messages = []
        st.success(f"Indexed {pdf.name}")

if "session_id" not in st.session_state:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()

# ---- chat history ----
st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---- new question ----
if question := st.chat_input("Ask about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API}/ask",
                    json={"session_id": st.session_state.session_id,
                          "question": question},
                    timeout=300,
                )
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Backend error: {e}")
                st.stop()

        with st.expander("Steps taken"):
            for s in data["steps"]:
                detail = f" ({s['docs']} docs)" if s["docs"] is not None else ""
                st.write(f"✅ **{s['node']}**{detail}")
        if data["used_web"]:
            st.caption("🌐 PDF wasn't enough, so web search was used.")
        st.markdown(data["answer"])
    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})