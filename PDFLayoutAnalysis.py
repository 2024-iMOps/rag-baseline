import os
import pymupdf
import json
import requests

from glob import glob
from PIL import Image

from bs4 import BeautifulSoup
from markdownify import markdownify as markdown

from dotenv import load_dotenv
load_dotenv()



def split_pdf(filepath, spl_data_path, batch_size=10):
    """
    입력 PDF를 여러 개의 작은 PDF 파일로 분할
    """

    # PDF 파일 열기
    input_pdf = pymupdf.open(filepath)
    num_pages = len(input_pdf)
    #print(f"총 페이지 수: {num_pages}")

    ret = []
    # PDF 분할
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page + batch_size, num_pages) - 1

        # 분할된 PDF 저장
        save_folder = os.path.splitext(filepath)[0]
        save_folder = os.path.basename(save_folder)
        #print(save_folder)
        save_path = f"{spl_data_path}/{save_folder}"
        #print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        output_file = f"{save_path}/{start_page:04d}_{end_page:04d}.pdf"
        #print(f"분할 PDF 생성: {output_file}")

        with pymupdf.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            ret.append(output_file)

    # 입력 PDF 파일 닫기
    input_pdf.close()
    return ret


class LayoutAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key

    def _upstage_layout_analysis(self, input_file):
        """
        레이아웃 분석 API 호출

        :param input_file: 분석할 PDF 파일 경로
        :param output_file: 분석 결과를 저장할 JSON 파일 경로
        """
        # API 요청 보내기
        response = requests.post(
            "https://api.upstage.ai/v1/document-ai/layout-analysis",
            headers={"Authorization": f"Bearer {self.api_key}"},
            data={"ocr": True},
            files={"document": open(input_file, "rb")},
        )

        # 응답 저장
        if response.status_code == 200:
            output_file = os.path.splitext(input_file)[0] + ".json"
            with open(output_file, "w") as f:
                json.dump(response.json(), f, ensure_ascii=False)
            return output_file
        else:
            raise ValueError(f"Error: {response.status_code}")

    def execute(self, input_file):
        return self._upstage_layout_analysis(input_file)


class PDFImageProcessor:
    """
    PDF 이미지 처리를 위한 클래스

    PDF 파일에서 이미지를 추출하고, HTML 및 Markdown 형식으로 변환하는 기능을 제공합니다.
    """

    def __init__(self, pdf_file):
        """
        PDFImageProcessor 클래스의 생성자

        :param pdf_file: 처리할 PDF 파일의 경로
        """
        self.pdf_file = pdf_file
        self.json_files = sorted(glob(os.path.splitext(pdf_file)[0] + "*.json"))
        self.output_folder = os.path.dirname(pdf_file)
        self.filename = os.path.splitext(os.path.basename(pdf_file))[0]

    @staticmethod
    def _load_json(json_file):
        """
        JSON 파일을 로드하는 정적 메서드

        :param json_file: 로드할 JSON 파일의 경로
        :return: JSON 데이터를 파이썬 객체로 변환한 결과
        """
        with open(json_file, "r") as f:
            return json.load(f)

    @staticmethod
    def _get_page_sizes(json_data):
        """
        각 페이지의 크기 정보를 추출하는 정적 메서드

        :param json_data: JSON 데이터
        :return: 페이지 번호를 키로, [너비, 높이]를 값으로 하는 딕셔너리
        """
        page_sizes = {}
        for page_element in json_data["metadata"]["pages"]:
            width = page_element["width"]
            height = page_element["height"]
            page_num = page_element["page"]
            page_sizes[page_num] = [width, height]
        return page_sizes

    def pdf_to_image(self, page_num, dpi=300):
        """
        PDF 파일의 특정 페이지를 이미지로 변환하는 메서드

        :param page_num: 변환할 페이지 번호 (1부터 시작)
        :param dpi: 이미지 해상도 (기본값: 300)
        :return: 변환된 이미지 객체
        """
        with pymupdf.open(self.pdf_file) as doc:
            page = doc[page_num - 1].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
        return page_img

    @staticmethod
    def normalize_coordinates(coordinates, output_page_size):
        """
        좌표를 정규화하는 정적 메서드

        :param coordinates: 원본 좌표 리스트
        :param output_page_size: 출력 페이지 크기 [너비, 높이]
        :return: 정규화된 좌표 (x1, y1, x2, y2)
        """
        x_values = [coord["x"] for coord in coordinates]
        y_values = [coord["y"] for coord in coordinates]
        x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)

        return (
            x1 / output_page_size[0],
            y1 / output_page_size[1],
            x2 / output_page_size[0],
            y2 / output_page_size[1],
        )

    @staticmethod
    def crop_image(img, coordinates, output_file):
        """
        이미지를 주어진 좌표에 따라 자르고 저장하는 정적 메서드

        :param img: 원본 이미지 객체
        :param coordinates: 정규화된 좌표 (x1, y1, x2, y2)
        :param output_file: 저장할 파일 경로
        """
        img_width, img_height = img.size
        x1, y1, x2, y2 = [
            int(coord * dim)
            for coord, dim in zip(coordinates, [img_width, img_height] * 2)
        ]
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save(output_file)

    def extract_images(self):
        """
        전체 이미지 처리 과정을 실행하는 메서드

        PDF에서 이미지를 추출하고, HTML 및 Markdown 파일을 생성합니다.
        """
        figure_count = {}  # 페이지별 figure 카운트를 저장하는 딕셔너리

        output_folder = self.output_folder
        os.makedirs(output_folder, exist_ok=True)

        print(f"Created Folder: {output_folder}")

        html_content = []  # HTML 내용을 저장할 리스트

        for json_file in self.json_files:
            json_data = self._load_json(json_file)
            page_sizes = self._get_page_sizes(json_data)

            # 파일 이름에서 페이지 범위 추출
            page_range = os.path.basename(json_file).split("_")[1:][0].split(".")[0]
            start_page = int(page_range[0])

            for element in json_data["elements"]:
                if element["category"] == "figure":
                    # 파일 내에서의 상대적인 페이지 번호 계산
                    relative_page = element["page"]
                    page_num = start_page + relative_page
                    coordinates = element["bounding_box"]
                    output_page_size = page_sizes[relative_page]
                    pdf_image = self.pdf_to_image(page_num)
                    normalized_coordinates = self.normalize_coordinates(
                        coordinates, output_page_size
                    )

                    # 페이지별 figure 카운트 관리
                    if page_num not in figure_count:
                        figure_count[page_num] = 1
                    else:
                        figure_count[page_num] += 1

                    # 출력 파일명 생성
                    output_file = os.path.join(
                        output_folder,
                        f"page_{page_num}_figure_{figure_count[page_num]}.png",
                    )

                    self.crop_image(pdf_image, normalized_coordinates, output_file)

                    # HTML에서 이미지 경로 업데이트
                    soup = BeautifulSoup(element["html"], "html.parser")
                    img_tag = soup.find("img")
                    if img_tag:
                        # 상대 경로로 변경
                        relative_path = os.path.relpath(output_file, output_folder)
                        img_tag["src"] = relative_path.replace("\\", "/")
                    element["html"] = str(soup)

                    print(f"Saved Image: {output_file}")

                html_content.append(element["html"])

        # HTML 파일 저장
        html_output_file = os.path.join(output_folder, f"{self.filename}.html")

        combined_html_content = "\n".join(html_content)
        soup = BeautifulSoup(combined_html_content, "html.parser")
        all_tags = set([tag.name for tag in soup.find_all()])
        html_tag_list = [tag for tag in list(all_tags) if tag not in ["br"]]

        with open(html_output_file, "w", encoding="utf-8") as f:
            f.write(combined_html_content)

        print(f"Saved HTML: {html_output_file}")

        # Markdown 파일 저장
        md_output_file = os.path.join(output_folder, f"{self.filename}.md")

        md_output = markdown(
            combined_html_content,
            convert=html_tag_list,
        )

        with open(md_output_file, "w", encoding="utf-8") as f:
            f.write(md_output)

        print(f"Saved Markdown: {md_output_file}")