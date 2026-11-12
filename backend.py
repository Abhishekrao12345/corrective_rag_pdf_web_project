"""
backend.py - FastAPI server: PDF indexing + the corrective RAG graph.
Run with:  uvicorn backend:app --reload
"""
import os
import tempfile
import uuid
from typing import List, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.embeddings import FastEmbedEmbeddings
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, UploadFile, File, HTTPException

load_dotenv()


MAX_RETRIES = 1     
MIN_GOOD_DOCS = 2    


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)  
web_tool = TavilySearch(max_results=3)                            
embeddings = FastEmbedEmbeddings()   

def build_retriever(pdf_path: str, k: int = 5):
    pages = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(pages)
    vectorstore = Chroma.from_documents(chunks, embeddings)  # in-memory
    return vectorstore.as_retriever(search_kwargs={"k": k})



class Grade(BaseModel):
    relevant: bool = Field(description="True if the excerpt helps answer the question")


grader = ChatPromptTemplate.from_messages([
    ("system", "Grade whether the excerpt is relevant to the question. "
               "Lenient on wording, strict on topic."),
    ("human", "Question: {question}\n\nExcerpt:\n{chunk}"),
]) | llm.with_structured_output(Grade)

rewriter = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question for semantic search over a document. "
               "Expand abbreviations, add key terms. Return only the query."),
    ("human", "{question}"),
]) | llm | StrOutputParser()

generator = ChatPromptTemplate.from_messages([
    ("system",
     """Answer only from the excerpts below.
- Cite PDF excerpts as (Page X) and web excerpts as (Web: url).
- If the excerpts don't contain the answer, say you couldn't find it.
- Say clearly when part of the answer comes from the web rather than the PDF.

Excerpts:
{context}"""),
    ("human", "{question}"),
]) | llm | StrOutputParser()


def format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        if d.metadata.get("source_type") == "web":
            label = f"[Web: {d.metadata.get('url')}]"
        else:
            label = f"[Page {d.metadata.get('page', 0) + 1}]"
        parts.append(f"{label}\n{d.page_content}")
    return "\n\n".join(parts)



class GraphState(TypedDict):
    question: str
    query: str
    documents: List[Document]
    generation: str
    retries: int
    used_web: bool


def initial_state(question: str) -> GraphState:
    return {"question": question, "query": question, "documents": [],
            "generation": "", "retries": 0, "used_web": False}



def build_graph(retriever):

    def retrieve(state):
        return {"documents": retriever.invoke(state["query"])}

    def grade_documents(state):
        good = [d for d in state["documents"]
                if grader.invoke({"question": state["question"],
                                  "chunk": d.page_content}).relevant]
        return {"documents": good}

    def rewrite_query(state):
        return {"query": rewriter.invoke({"question": state["question"]}),
                "retries": state["retries"] + 1}

    def web_search(state):
        res = web_tool.invoke({"query": state["question"]})
        web_docs = [
            Document(page_content=r["content"],
                     metadata={"source_type": "web", "url": r["url"]})
            for r in res.get("results", [])
        ]
        # keep any good PDF chunks we found, add the web ones
        return {"documents": state["documents"] + web_docs, "used_web": True}

    def generate(state):
        if not state["documents"]:
            return {"generation": "I couldn't find this in the PDF or on the web."}
        return {"generation": generator.invoke({
            "context": format_docs(state["documents"]),
            "question": state["question"],
        })}

    def decide_next(state):
        if len(state["documents"]) >= MIN_GOOD_DOCS:
            return "generate"
        if state["retries"] < MAX_RETRIES:
            return "rewrite_query"
        return "web_search"

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", decide_next, {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search",
    })
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)
    return g.compile()



app = FastAPI(title="Corrective RAG API")

GRAPHS = {}  


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Receive a PDF, index it, return a session_id."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file.file.read())
        path = f.name
    try:
        graph = build_graph(build_retriever(path))
    finally:
        os.remove(path)
    session_id = str(uuid.uuid4())
    GRAPHS[session_id] = graph
    return {"session_id": session_id, "filename": file.filename}


@app.post("/ask")
def ask(req: AskRequest):
    """Run the corrective RAG graph and return the answer + the steps taken."""
    graph = GRAPHS.get(req.session_id)
    if graph is None:
        raise HTTPException(404, "Unknown session. Upload a PDF first.")

    steps, final = [], {}
    for step in graph.stream(initial_state(req.question)):
        for node, out in step.items():
            steps.append({"node": node,
                          "docs": len(out["documents"]) if "documents" in out else None})
            final.update(out)

    return {"answer": final.get("generation", "No answer generated."),
            "used_web": final.get("used_web", False),
            "steps": steps}


@app.get("/health")
def health():
    return {"status": "ok"}