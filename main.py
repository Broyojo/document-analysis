import base64
from io import BytesIO
from openai import OpenAI
from pdf2image import convert_from_path
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

client = OpenAI()
images = convert_from_path("./doc.pdf")


def encode_base64(img):
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes).decode("utf-8")
    return im_b64


for chunk in client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are an assistant which extracts information from documents",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_base64(image)}"
                    },
                }
                for image in images
            ],
        },
    ],
    stream=True,
):
    print(chunk.choices[0].delta.content, end="", flush=True)
