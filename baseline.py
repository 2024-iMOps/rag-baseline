import os
import re

import numpy as np
import pandas as pd

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
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import PromptTemplate

from langchain_core.documents import Document

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from huggingface_hub import login
#login("your-key")

import warnings
warnings.filterwarnings('ignore')

# seed
import torch
import random
from transformers import set_seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
set_seed(seed)


def format_docs(docs):
    global references
    references = docs
    context = ""
    for doc in docs:
        context += "\n\n" + doc.page_content
    return context


# Embedding
model_path="BAAI/bge-multilingual-gemma2" # BAAI/bge-multilingual-gemma2 intfloat/multilingual-e5-base jhgan/ko-sroberta-multitask"
model_kwargs={"device": "cuda"} # cuda cpu
encode_kwargs={"normalize_embeddings": True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
torch.cuda.empty_cache()


# Text Splitter
f_path = "/workspace/rag-baseline/data-pdf-split/card/0_iM Social Worker카드" # 각자 데이터로 대체
md_files = [pdf for pdf in os.listdir(f_path) if pdf.endswith(".md")]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=384,
    chunk_overlap=32,
    # length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n\n",
        "\n\n",
        # "\n",
        " ",
        ".",
    ],
)

md_splits = []

for md_file in tqdm(md_files):

    with open(f"{f_path}/{md_file}", "r", encoding="utf-8") as file:
        file_read = file.read()

    temp = splitter.split_text(file_read)
    md_splits.append(temp)


# Text to Doc
docs = []
for i in tqdm(range(len(md_splits))):
    for j, split_text in enumerate(md_splits[i]):
        fp = f"{f_path}/{md_files[i]}"
        doc = Document(
            page_content=split_text,
            metadata={
                'FilePath': fp,
                'DocName': os.path.basename(f_path),
                'DocPage': i
            }
        )
        docs.append(doc)


# Retriever
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


# Model
model_id = "NCSOFT/Llama-VARCO-8B-Instruct" # rtzr/ko-gemma-2-9b-it
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
#prompt = hub.pull("rlm/rag-prompt")
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


# QA pair (각자 csv file로 대체)
questions = [
    "할부 결제가 가능한 카드는 무엇인가요?",
    "신용카드의 특성은?",
    "신용카드 이용이 제한되는 가맹점을 알려주세요.",
    "해외 서비스의 수수료율은 어떻게 되나요?",
    "기한의 이익은 무엇이고, 기한의 이익이 상실되는 주요 사유는 무엇인가요?",
    "신용카드를 해지하고 싶어요.",
    "개인신용평가대응원이 뭐야?",
    "신용카드 이용대금을 연체하면 어떻게 돼?",
    "아이엠뱅크 고객센터 전화번호를 알려줘.",
    "자동 납부 업무 시간이 지나게 되면 결제를 어떻게 해?",
    "커피 할인 대상이 어떻게 돼?",
    "해외에서 신용카드를 이용할 때 청구 금액을 알려줘.",
]

answers = [
    "할부 결제가 가능한 카드는 신용카드입니다.",
    "신용카드의 특성은 회원의 신용을 담보로 신용카드 가맹점에서 상품 등을 현금의 즉시 지불 없이 구입할 수 있는 증표입니다.",
    "신용카드 이용이 제한되는 가맹점은 카지노, 경마, 경정, 경륜장, 복권방, 해외가상화폐거래소 등이 있습니다.",
    "해외 서비스의 수수료율은 0.25% 입니다.",
    "기한의 이익은 기한의 존재로 말미암아 당사자가 받는 이익으로, 기한의 이익이 상실되는 주요 사유는 신용카드 거래와 관련하여 허위, 위변조 또는 고의로 부실자료를 제출하여 카드사의 채권보전에 중대한 손실을 유발한 때, 신용카드 이용대금(단기카드대출(현금서비스) 포함) 또는 다른 금융기관에 대한 채무를 연체한 경우, 할부금을 연속하여 2회 이상 연체하고, 그 연체한 금액이 총 할부금액의 1/10을 초과하는 경우, 다른 채무로 인하여 압류, 경매, 기타 강제 집행을 당한 경우 등이 있습니다.",
    "회원은 인터넷 홈페이지, 은행앱, 고객센터를 통해 신용카드의 해지를 신청할 수 있습니다. (기업회원 제외)",
    "개인신용평가대응권이란 개인인 금융소비자가 자동화평가에 따른 개인신용평가 결과 및 주요기준 등의 설명과 자동화평가 결과의 산출에 유리하다고 판단되는 정보를 제출 또는 기초정보의 정정ㆍ삭제ㆍ재산출을 요구할 수 있는 권리(신용정보의 이용 및 보호에 관한 법률 제36조의2)를 말합니다.",
    "원리금에 연체이자율이 적용되고 금융거래가 제약되며 신용점수 등이 하락하는 불이익이 발생할 수 있습니다.",
    "아이엠뱅크 고객센터 전화번호는 1566-5050 입니다.",
    "자동납부 업무 마감시간 이후에는 당행 홈페이지, 모바일웹 및 앱에서 즉시 결제 또는 입금전용(가상)계좌 입금(송금납부)을 통해 당일 결제가 가능합니다.",
    "커피 할인 대상은 스타벅스, 투썸플레이스, 커피빈, 폴바셋, 할리스, 이디야커피, 블루보틀 입니다.",
    "해외에서 신용카드를 이용할 시 청구 금액은 (거래미화금액 * 전신환매도율) + 국제브랜드수수료 + 해외서비스수수료 입니다.",
]

docname = ["0_iM Social Worker카드"] * 12
docpages = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


data = {
    "Question": questions,
    "Answer": answers,
    "DocName": docname,
    "DocPage": docpages
}
qa_pair_df = pd.DataFrame(data)


# Inference
results = []

idx = 0
for _, row in qa_pair_df.iterrows():
    
    question = row['Question']
    retriever = faiss_retriever

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 답변 추론
    # print(f"Task: {source}")
    print(f"[{idx+1}/{qa_pair_df.shape[0]}]", "--------"*6)
    print(f"Question: {question}")
    response = rag_chain.invoke(question).strip()

    print(f"Answer: {response}")

    ref_text = [reference.page_content for reference in references]
    ref_path = [reference.metadata["FilePath"] for reference in references]
    ref_name = [reference.metadata["DocName"] for reference in references]
    ref_page = [reference.metadata["DocPage"] for reference in references]

    # 결과 저장
    results.append({
        "Question": question,
        "Answer": response,
        "Reference": ref_text,
        "FilePath": ref_path,
        "DocName": ref_name,
        "DocPage": ref_page,
    })

    idx += 1

    torch.cuda.empty_cache()