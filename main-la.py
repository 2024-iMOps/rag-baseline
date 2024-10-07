import os
import natsort

from glob import glob
from tqdm import tqdm

from PDFLayoutAnalysis import LayoutAnalyzer, PDFImageProcessor, split_pdf

from dotenv import load_dotenv
load_dotenv()


# 업스테이지 API key 발급 후 .env 파일에 기재
analyzer = LayoutAnalyzer(os.environ.get("UPSTAGE_API_KEY"))


# 카테고리별 설정
category = "card"
BATCH_SIZE = 4 # 페이지 나누는 기준: 10페이지의 문서일 경우, 4-4-2로 나뉨


# 데이터 저장 경로 설정
base_path = "/workspace/rag-baseline"
org_data_path = f"{base_path}/data/{category}" # 원본 데이터 경로
spl_data_path = f"{base_path}/data-pdf-split/{category}" # 레이아웃 분석 후 저장될 데이터 경로


# 원본 데이터 리스트 생성
org_data_list = [pdf for pdf in os.listdir(org_data_path) if pdf.endswith(".pdf")]
org_data_list = natsort.natsorted(org_data_list)


# 레이아웃 분석 데이터 경로 생성
if not os.path.exists(spl_data_path):
    os.makedirs(spl_data_path)


# 레이아웃 분석 시작
for temp in tqdm(org_data_list):

    folder = os.path.splitext(temp)[0]
    #print(folder)
    json_path = f"{spl_data_path}/{folder}/*.json"
    json_files = glob(json_path)

    # json 파일 없는 경우 (레이아웃 분석 데이터 경로에 저장됨)
    if not json_files:

        split_files = split_pdf(f"{org_data_path}/{temp}", spl_data_path, BATCH_SIZE)

        analyzed_files = []
        for file in split_files:
            analyzed_files.append(analyzer.execute(file))

    # json 파일 있는 경우
    else:
        print("Layout Analysis Already Exists.")


# 레이아웃 분석 데이터 리스트 생성
spl_data_list = natsort.natsorted(os.listdir(spl_data_path))

# HTML to Markdown
for temp in tqdm(spl_data_list):

    folder = os.path.splitext(temp)[0]

    pdf_path = f"{spl_data_path}/{folder}/*.pdf"
    pdf_files = glob(pdf_path)

    for pdf_file in pdf_files:
        
        md_path = f"{spl_data_path}/{folder}/*.md"
        md_files = glob(md_path)

        # md 파일 있는 경우
        if not md_files:
            image_processor = PDFImageProcessor(pdf_file)
            image_processor.extract_images()

        # md 파일 없는 경우
        else: 
            print("HTML and Markdown Already Exists.")