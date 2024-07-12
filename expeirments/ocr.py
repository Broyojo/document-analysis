import os
from PIL import Image, ImageDraw
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat
import concurrent.futures

# billing: https://portal.azure.com/#view/Microsoft_Azure_GTM/BillingAccountMenuBlade/~/Overview/billingAccountId/%2Fproviders%2FMicrosoft.Billing%2FbillingAccounts%2F3f6e673c-aaeb-4c82-a5cf-07f492ecd074%3Aca8c8f47-c534-48a4-995e-a95d4dceb258_2019-05-31/accountType/Individual

endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)


def ocr(image_path: str, markdown: bool = False):
    with open(image_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-read",
            analyze_request=f,
            content_type="application/octet-stream",
            output_content_format=ContentFormat.MARKDOWN if markdown else None,
        )
    result = poller.result()
    return result


def parallel_ocr(image_paths: list[str], max_workers: int = 20, markdown: bool = False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda x: (x[0], ocr(x[1], markdown=markdown)),
                enumerate(image_paths),
            )
        )

    results.sort(key=lambda x: x[0])
    results = [result[1] for result in results]
    return results


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


if __name__ == "__main__":
    # Your existing code to get the document analysis result

    PARALLEL = True

    if PARALLEL:
        from pathlib import Path
        import time

        files = Path("/mnt/ssd/images/").glob("ffbl0226_p*.jpg")
        image_paths = [str(file) for file in files]
        print(image_paths)

        start = time.time()
        results = parallel_ocr(image_paths)
        print(
            f"Time taken: {time.time() - start:.2f} seconds | ~{len(image_paths) / (time.time() - start):.2f} seconds per image"
        )

        for result in results:
            print(result["content"])
            print("\n\n====================================\n")

    else:

        image_path = "/mnt/ssd/images/ffbl0226_p2.jpg"

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

        print(words)

        concated = ""
        for word in words:
            concated += word["content"] + " "

        # note: markdown seems to be better for structured output, but the ocr quality is the same as the text output
        with open("ocr_text_output.txt", "w") as f:
            f.write(concated)

        # Draw bounding boxes
        draw_bounding_boxes(image_path, words, "boundingbox_ocr.jpg")
