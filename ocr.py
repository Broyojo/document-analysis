import os
from PIL import Image, ImageDraw
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat

# billing: https://portal.azure.com/#view/Microsoft_Azure_GTM/BillingAccountMenuBlade/~/Overview/billingAccountId/%2Fproviders%2FMicrosoft.Billing%2FbillingAccounts%2F3f6e673c-aaeb-4c82-a5cf-07f492ecd074%3Aca8c8f47-c534-48a4-995e-a95d4dceb258_2019-05-31/accountType/Individual


def draw_bounding_boxes(image_path, words, output_path):
    # Open the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        # Draw rectangles for each word
        for word in words:
            box = word["polygon"]
            # Ensure the coordinates are in the correct order (left, top, right, bottom)
            left = min(box[0], box[6])
            top = min(box[1], box[3])
            right = max(box[2], box[4])
            bottom = max(box[5], box[7])
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Save the new image
        img.save(output_path)
        print(f"Image with bounding boxes saved as {output_path}")


# Your existing code to get the document analysis result
endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

image_path = "/mnt/ssd/images/ffcn0226_p5.jpg"

with open(image_path, "rb") as f:
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout",
        analyze_request=f,
        content_type="application/octet-stream",
        output_content_format=ContentFormat.MARKDOWN,
    )
result = poller.result()

with open("ocr_markdown_output.md", "w") as f:
    f.write(result["content"])

# Get words from the first page (assuming single-page document)
words = result.pages[0].words

# Draw bounding boxes
draw_bounding_boxes(image_path, words, "boundingbox_ocr.jpg")
