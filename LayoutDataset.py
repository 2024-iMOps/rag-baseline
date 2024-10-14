import os
import json
import requests
import fitz

from tqdm import tqdm
from bs4 import BeautifulSoup
from markdownify import markdownify as markdown

from dotenv import load_dotenv
load_dotenv()

class UpstageDocParse:
    def __init__(self):
        """
        Upstage Document Parse 활용
        """
        self.api_key = os.environ.get("UPSTAGE_API_KEY")
        self.url = "https://api.upstage.ai/v1/document-ai/document-parse"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def save_json(self, filename, save_path):
        """
        API 반환값을 JSON 형식으로 저장
        
        filename: pdf 이름<br>
        save_path: json 파일을 저장할 경로
        """
        files = {"document": open(filename, "rb")}
        response = requests.post(self.url, headers=self.headers, data={"ocr": True}, files=files)
        result = response.json()

        json_file_path = os.path.join(save_path, "parse_results.json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        # print("Save JSON file")

    def save_html(self, json_file_path, save_path):
        """
        JSON에서 HTML 요소 추출
        
        json_file_path: 불러올 json 파일 경로<br>
        save_path: html 파일을 저장할 경로
        """
        with open(json_file_path, "r") as f:
            json_data = json.load(f)

        html_file_path = os.path.join(save_path, "parse_results.html")
        with open(html_file_path, "w", encoding="utf-8") as f:
            for element in json_data["elements"]:
                f.write(element["content"]["html"] + "\n")

        # print("Save HTML file")

    def save_md(self, html_file_path, save_path):
        """
        HTML 형식을 Markdown으로 변환
        
        html_file_path: 불러올 html 파일 경로<br>
        save_path: markdown 파일을 저장할 경로
        """
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        soup = BeautifulSoup(html_data, "html.parser")
        all_tags = set([tag.name for tag in soup.find_all()])
        html_tag_list = [tag for tag in list(all_tags) if tag not in ["br"]]

        md_file_path = os.path.join(save_path, "parse_results.md")
        md_output = markdown(
            html_data,
            convert=html_tag_list,
        )

        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_output)

        # print("Save MD file")

def main():
    category = "card"
    data_dir = "/workspace/rag-baseline/data"
    save_base_path = "/workspace/rag-baseline/data-parse"

    doc_parser = UpstageDocParse()

    pdf_files = [f for f in os.listdir(os.path.join(data_dir, category)) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    with tqdm(total=total_files, desc="Parse PDF files") as pbar:
        for pdf_name in pdf_files:
            save_path = os.path.join(save_base_path, category, os.path.splitext(pdf_name)[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            filename = os.path.join(data_dir, category, pdf_name)

            doc_parser.save_json(filename, save_path)
            doc_parser.save_html(os.path.join(save_path, "parse_results.json"), save_path)
            doc_parser.save_md(os.path.join(save_path, "parse_results.html"), save_path)

            pbar.update(1)

if __name__ == "__main__":
    main()