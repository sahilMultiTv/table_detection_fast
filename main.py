import os
import json
import time
import shutil
import requests
import base64
import replicate
import urllib3
import uuid
import logging
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
import fitz
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import urllib.parse

# Load environment variables
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
replicate_api_token = os.getenv('REPLICATE_API_TOKEN')
imgbb_api_key = os.getenv('IMGBB_API_KEY')
azure_storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
azure_container_name = os.getenv('AZURE_CONTAINER_NAME')

app = FastAPI()

HORIZONTAL_THRESHOLD = 20  
VERTICAL_THRESHOLD = 30   


def rename_image_file(file_name):
    # Remove the .pdf from the filename
    new_name = file_name.replace('.pdf', '')
    
    return new_name


def extract_path_from_url(url):
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path

    # Remove the initial '/bynd-pdfs' and the trailing '/RAW/SPIL-AR2022-23-Complete-Annual-Report.pdf'
    parts = path.split('/')
    if len(parts) > 3:
        base_path = '/'.join(parts[2:-2])  # Extract parts between '/bynd-pdfs/' and '/RAW'
    else:
        base_path = ''  # Handle edge cases if there are fewer parts

    return base_path


def upload_image_to_blob(file_path, container_name, connection_string, base_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_name = f"{base_path}/processed/table/images/{rename_image_file(file_path)}" 
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"Uploaded image to blob: {blob_client.url}")
        return blob_client.url
    except Exception as e:
        logger.error(f"Failed to upload image to blob: {str(e)}")
        raise

def pdf_to_image_url(pdf_path, dpi=300, upload_to_blob=True, base_path=None, img_page_num=0):
    image_urls = []

    try:
        with fitz.open(pdf_path) as doc:
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            for page_num, page in enumerate(doc):
                logger.info(f"Processing page number: {page_num}")
                image = page.get_pixmap(matrix=matrix)
                img_path = f"image_{img_page_num}.png"
                image.save(img_path)

                if upload_to_blob:
                    if base_path is None:
                        raise ValueError("Base path must be provided for blob upload.")
                    img_url = upload_image_to_blob(img_path, azure_container_name, azure_storage_connection_string, base_path)
                    image_urls.append(img_url)
                else:
                    with open(img_path, "rb") as image_file:
                        img_byte_arr = image_file.read()
                        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
                        image_urls.append(base64_image)

                os.remove(img_path)
        logger.info(f"Image URLs: {image_urls}")
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {str(e)}")
        raise

    return image_urls

def split_pdf(pdf_path, output_folder="./splitPDF"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_number in range(len(pdf_reader.pages)):
                pdf_writer = PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[page_number])

                output_file_path = os.path.join(output_folder, f"{page_number + 1}.pdf")

                with open(output_file_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
        logger.info(f"PDF split into {len(os.listdir(output_folder))} pages.")
    except Exception as e:
        logger.error(f"Failed to split PDF: {str(e)}")
        raise

def numerical_sort(value):
    return int(value.split('_')[-1].split('.')[0])

def add_element_to_json_file_with_page_num(type, coordinates):
    full_data = {}

    for i in coordinates:
        full_data[i[1]] = []

        for j in i[0]:
            table_dic = {}
            table_dic['coordinates'] = j
            table_dic[f'{type}_id'] = str(uuid.uuid4())
            full_data[i[1]].append(table_dic)

    file_path = "bounding-box-tables.json" if type == 'tables' else "bounding-box-images.json"

    try:
        with open(file_path, 'w') as file:
            json.dump(full_data, file, indent=4)
        logger.info(f"Updated JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to update JSON file: {str(e)}")
        raise

def extract_element_coords(filePath, pdf_url):
    try:
        split_pdf(filePath)
        table_info = []
        image_info = []

        base_path = extract_path_from_url(pdf_url)

        for j in sorted(os.listdir("splitPDF"), key=numerical_sort):
            image_coordinates = []
            doc = fitz.open(f"splitPDF/{j}")
            page = doc[0]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                bbox = page.get_image_bbox(img)
                image_coordinates.append([bbox.x0, bbox.y0, bbox.x1, bbox.y1])

            image_info.append([image_coordinates, os.path.splitext(os.path.basename(j))[0]])
            doc.close()

            try:
                image_urls = pdf_to_image_url(f"splitPDF/{j}", base_path=base_path, img_page_num=j)
                logger.info(f"Image URLs: {image_urls}")
                for image_url in image_urls:
                    logger.info(f"Processing image URL: {image_url}")
                    output = replicate.run("prtm1908/bynd-table-extractor:5763df0014acf048ad8db87babcee2be25d475c541dfb5e2cf33268aa8a806e1", input={"image": image_url})
                    logger.info(f"Model output: {output}")

                    for a in range(len(output['boxes'])):
                        for b in range(len(output['boxes'][a])):
                            output['boxes'][a][b] /= 4.17

                            if b < 2:
                                output['boxes'][a][b] -= 5
                            else:
                                output['boxes'][a][b] += 5

                    table_info.append([output['boxes'], os.path.splitext(os.path.basename(j))[0]])
            except Exception as e:
                logger.error(f"Error processing image {j}: {str(e)}")

        add_element_to_json_file_with_page_num("tables", table_info)
        add_element_to_json_file_with_page_num("images", image_info)

        shutil.rmtree("splitPDF")
    except Exception as e:
        logger.error(f"Error extracting element coordinates: {str(e)}")
        raise

def is_encapsulating(outer, inner, padding=15):
    outer_x0, outer_y0, outer_x1, outer_y1 = outer['coordinates']
    inner_x0, inner_y0, inner_x1, inner_y1 = inner['coordinates']

    return (outer_x0 - padding <= inner_x0 and
            outer_x1 + padding >= inner_x1 and
            outer_y0 - padding <= inner_y0 and
            outer_y1 + padding >= inner_y1)

def process_tables(data):
    for key, tables in data.items():
        to_remove = set()

        for i, outer in enumerate(tables):
            for j, inner in enumerate(tables):
                if i != j and is_encapsulating(outer, inner):
                    to_remove.add(j)

        data[key] = [table for i, table in enumerate(tables) if i not in to_remove]

    return data

def merge_tables(table1, table2):
    x_min = min(table1['coordinates'][0], table2['coordinates'][0])
    y_min = min(table1['coordinates'][1], table2['coordinates'][1])
    x_max = max(table1['coordinates'][2], table2['coordinates'][2])
    y_max = max(table1['coordinates'][3], table2['coordinates'][3])

    return {
        'coordinates': [x_min, y_min, x_max, y_max]
    }

def should_merge(table1, table2):
    x1, y1, x2, y2 = table1['coordinates']
    x3, y3, x4, y4 = table2['coordinates']

    horizontal_distance = min(abs(x2 - x3), abs(x1 - x4))
    vertical_distance = min(abs(y2 - y3), abs(y1 - y4))

    return (horizontal_distance <= HORIZONTAL_THRESHOLD and
            vertical_distance <= VERTICAL_THRESHOLD)

def merge_close_tables(data):
    for key, page in data.items():
        if len(data[key]) < 2:
            continue

        i = 0
        while i < len(data[key]):
            j = i + 1
            while j < len(data[key]):
                if should_merge(data[key][i], data[key][j]):
                    merged_table = merge_tables(data[key][i], data[key][j])
                    data[key][i] = merged_table
                    del data[key][j]
                else:
                    j += 1
            i += 1

def process_and_save_to_blob(pdf_url: str):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        parsed_url = urllib.parse.urlparse(pdf_url)
        path = parsed_url.path

        file_name_with_ext = os.path.basename(path)
        file_name, _ = os.path.splitext(file_name_with_ext)

        pdf_filename = f"temp_{uuid.uuid4()}.pdf"
        with open(pdf_filename, 'wb') as file:
            file.write(response.content)

        blob_name_table = f"{extract_path_from_url(pdf_url)}/processed/table/bounding-box-tables.json"   
        logger.info(f"Table Blob name: {blob_name_table}")



        blob_name_image = f"{extract_path_from_url(pdf_url)}/processed/table/bounding-box-images.json"
        logger.info(f"Table Blob name: {blob_name_table}")


        extract_element_coords(pdf_filename, pdf_url)

        with open('bounding-box-tables.json', 'r') as f:
            table_data = json.load(f)

        with open('bounding-box-images.json', 'r') as f:
            image_data = json.load(f)    

        table_data = process_tables(table_data)
        merge_close_tables(table_data)

        upload_to_blob(table_data, blob_name_table)
        upload_to_blob(image_data, blob_name_image)

        os.remove(pdf_filename)
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"Error in background task: {str(e)}")
        raise

def upload_to_blob(data, blob_name):
    container_name = os.getenv("AZURE_CONTAINER_NAME")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(blob_name)
    logger.info("Uploading processed data to blob.")
    blob_client.upload_blob(json.dumps(data), overwrite=True)


@app.post("/extract-coords/")
async def extract_coords(background_tasks: BackgroundTasks, pdf_url: str = Form(...)):
    print('started')
    background_tasks.add_task(process_and_save_to_blob, pdf_url)
    return JSONResponse(content={"message": "Processing started in the background"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



