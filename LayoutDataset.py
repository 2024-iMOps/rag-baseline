# 참고: https://developers.upstage.ai/docs/apis/document-parse

import os
import json
import requests
import pymupdf

from dotenv import load_dotenv
load_dotenv()

# 카테고리 및 데이터
category = "card"
pdfdata = "0_iM Social Worker카드.pdf"

# 데이터 저장 경로
save_path = f"/workspace/rag-baseline/data-docparse/{category}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# PDF 파일
filename = f"/workspace/rag-baseline/data/{category}/{pdfdata}"

# 업스테이지 API 설정
api_key = os.environ.get("UPSTAGE_API_KEY")
url = "https://api.upstage.ai/v1/document-ai/document-parse"
headers = {"Authorization": f"Bearer {api_key}"}
files = {"document": open(filename, "rb")}

# API 호출 및 결과 JSON 저장
response = requests.post(url, headers=headers, data={"ocr": True}, files=files)
result = response.json()
with open(f"{save_path}/parse_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)