# 참고: https://developers.upstage.ai/docs/apis/document-parse
# 페이지에 이미지가 있을 때 저장하는 코드 추가 필요

import os
import json
import requests
import fitz

from tqdm import tqdm
from bs4 import BeautifulSoup
from markdownify import markdownify as markdown

from dotenv import load_dotenv
load_dotenv()


# 카테고리 및 데이터
category = "card"
pdf_name = "0_iM Social Worker카드"
save_name = "parse_results"
save_path = f"/workspace/rag-baseline/data-parse/{category}/{pdf_name}"

json_file_path = f"{save_path}/{save_name}.json"
html_file_path = f"{save_path}/{save_name}.html"
md_file_path = f"{save_path}/{save_name}.md"
# image_path = f"{save_path}/images"


# 데이터 저장 경로
if not os.path.exists(save_path):
    os.makedirs(save_path)
# if not os.path.exists(image_path):
#     os.makedirs(image_path)


# PDF 파일
filename = f"/workspace/rag-baseline/data/{category}/{pdf_name}.pdf"


# 업스테이지 API 설정
api_key = os.environ.get("UPSTAGE_API_KEY")
url = "https://api.upstage.ai/v1/document-ai/document-parse"
headers = {"Authorization": f"Bearer {api_key}"}
files = {"document": open(filename, "rb")}


# JSON 저장
response = requests.post(url, headers=headers, data={"ocr": True}, files=files)
result = response.json()

with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("Save JSON file")


# HTML 저장
with open(json_file_path, "r") as f:
    json_data = json.load(f)

with open(html_file_path, "w", encoding="utf-8") as f:
    for element in json_data["elements"]:
        f.write(element["content"]["html"] + "\n")

print("Save HTML file")


# MD 저장
with open(html_file_path, "r", encoding="utf-8") as f:
    html_data = f.read()
soup = BeautifulSoup(html_data, "html.parser")
all_tags = set([tag.name for tag in soup.find_all()])
html_tag_list = [tag for tag in list(all_tags) if tag not in ["br"]]

md_output = markdown(
    html_data,
    convert=html_tag_list,
)

with open(md_file_path, "w", encoding="utf-8") as f:
    f.write(md_output)

print("Save MD file")