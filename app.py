import os

os.system('pip install auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl')

import streamlit as st
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from streamlit_chat import message

# Check if device is available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize everything in session state to avoid reloading
if "initialized" not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    
    # Load PDF
    loader = PyPDFLoader("Medical_Book.pdf")
    docs = loader.load()

    # Initialize embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)

    # Create Chroma vectorstore
    db = Chroma.from_documents(texts, embeddings, persist_directory="db")

    # Load model and tokenizer
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        revision="gptq-4bit-128g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    # Set system prompt
    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    """.strip()

    def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f"""
        [INST] <>
        {system_prompt}
        <>
        {prompt} [/INST]
        """.strip()

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    template = generate_prompt(
        """
        {context}
        Question: {question}
        """,
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    st.session_state.qa_chain = qa_chain
    st.session_state.initialized = True

st.title("Medical Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history using streamlit-chat
for i, chat in enumerate(st.session_state.history):
    message(chat['question'], is_user=True, key=f"user_{i}")
    message(chat['answer'], key=f"bot_{i}")

user_input = st.chat_input(placeholder="Ask a question:", key="input")

# if st.button("Generate"):
if user_input:
    result = st.session_state.qa_chain(user_input)
    answer = result["result"]
    st.session_state.history.append({"question": user_input, "answer": answer})
    st.experimental_rerun()
    