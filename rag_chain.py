import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()

CHROMA_DIR = "./chroma_db"

SYSTEM_PROMPT = """You are KIITBot, a helpful and friendly assistant for students of Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar.

You answer questions about:
- University info, rankings, faculty, and contacts
- Exam rules, grading system, and attendance policy
- Fees, scholarships, and finance
- Admissions and the KIITEE process
- Academic calendar and exam schedules for 2025-26
- Course curriculum, credits, and electives
- Campus life, hostels, sports, and clubs
- Placements and internships
- KIIT NEXUS — the student technical community
- KIIT Quest — the exam preparation app by KIIT NEXUS

STRICT RULES:
1. Answer ONLY using the context provided below. Do not use any outside knowledge.
2. If the answer is not in the context, say exactly: "I don't have that information right now. Please contact the relevant office." Then suggest the correct contact from KIIT (e.g. compliance.cse@kiit.ac.in).
3. Be concise, friendly, and direct. Use bullet points for lists.
4. Never make up exam dates, fees, names, or rules.
5. If a student asks something outside KIIT topics, politely say you can only help with KIIT-related queries.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

def build_chain():
    # Load embeddings — same model used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Load ChromaDB from disk
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Retriever — fetch top 4 most relevant chunks per query
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Groq LLM — free, fast, no credit card
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=600,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Memory — keeps last 5 exchanges so follow-up questions work
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Custom prompt
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT
    )

    # Build the full RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )

    return chain