import os
import re
import numpy as np
import pandas as pd
import torch
import random
import streamlit as st
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from langchain import hub
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import EnsembleRetriever

# Set random seed
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def format_docs(docs):
    global references
    references = docs
    context = ""
    for doc in docs:
        context += "\n\n" + doc.page_content
    return context

# Streamlit setup
st.title("RAG Chatbot Demo")

# Select a folder containing markdown files
folder_path = st.selectbox("Select Folder", [
    "/workspace/rag-baseline/data-pdf-split/card/0_iM Social Worker\uce74\ub4dc"
])

# List markdown files in selected folder
md_files = [pdf for pdf in os.listdir(folder_path) if pdf.endswith(".md")]

# Display markdown files
st.write("Files found:")
st.write(md_files)

# Embedding Model
model_path = "BAAI/bge-multilingual-gemma2"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
torch.cuda.empty_cache()

# Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=384,
    chunk_overlap=32,
    is_separator_regex=False,
    separators=[
        "\n\n\n",
        "\n\n",
        " ",
        ".",
    ],
)

md_splits = []
for md_file in tqdm(md_files):
    with open(f"{folder_path}/{md_file}", "r", encoding="utf-8") as file:
        file_read = file.read()
    temp = splitter.split_text(file_read)
    md_splits.append(temp)

# Text to Document Conversion
docs = []
for i in tqdm(range(len(md_splits))):
    for j, split_text in enumerate(md_splits[i]):
        fp = f"{folder_path}/{md_files[i]}"
        doc = Document(
            page_content=split_text,
            metadata={
                'FilePath': fp,
                'DocName': os.path.basename(folder_path),
                'DocPage': i
            }
        )
        docs.append(doc)

# FAISS Retriever
faiss_db = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)
faiss_retriever = faiss_db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10,
    }
)

# Language Model
model_id = "NCSOFT/Llama-VARCO-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

eos_token_id = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.1,
    do_sample=True,
    return_full_text=False,
    max_new_tokens=256,
    eos_token_id=eos_token_id,
)

hf = HuggingFacePipeline(pipeline=text_generation_pipeline)
llm = hf
torch.cuda.empty_cache()

# Prompt
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an assistant for question-answering tasks.<|eot_id|><|start_header_id|>user<|end_header_id|>

Use the following pieces of retrieved context to answer the question. \
Provide an answer between one to three sentences and keep it concise. \
Answer only the question being asked. \
Write the answer in Korean.

context: {context}

question: {question}

answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
prompt = PromptTemplate.from_template(template)

# Chatbot Functionality
st.header("Chatbot Interaction")
question = st.text_input("Ask a question:")
if st.button("Submit") and question:
    retriever = faiss_retriever
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question).strip()
    st.write("Answer:")
    st.write(response)
    if references:
        st.write("Reference Document:")
        for ref in references:
            st.write(ref.metadata['FilePath'])