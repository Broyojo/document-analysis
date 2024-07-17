import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import concurrent.futures
from fuzzywuzzy import fuzz
from PIL import Image, ImageDraw

# billing: https://portal.azure.com/#view/Microsoft_Azure_GTM/BillingAccountMenuBlade/~/Overview/billingAccountId/%2Fproviders%2FMicrosoft.Billing%2FbillingAccounts%2F3f6e673c-aaeb-4c82-a5cf-07f492ecd074%3Aca8c8f47-c534-48a4-995e-a95d4dceb258_2019-05-31/accountType/Individual

endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)


def ocr(image_paths: list[str]) -> list[AnalyzeResult]:
    def ocr_single(image_path: str) -> AnalyzeResult:
        with open(image_path, "rb") as f:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-read",
                analyze_request=f,
                content_type="application/octet-stream",
            )
        result = poller.result()
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(
            executor.map(
                lambda x: (x[0], ocr_single(x[1])),
                enumerate(image_paths),
            )
        )

    results.sort(key=lambda x: x[0])
    results = [result[1] for result in results]
    return results


def find_sentence_words(quote, page_words, match_threshold=30):
    quote_words = quote.lower().split()
    matched_words = []
    i = 0
    for page_word in page_words:
        if i >= len(quote_words):
            break
        if fuzz.ratio(quote_words[i], page_word["content"].lower()) > match_threshold:
            matched_words.append(page_word)
            i += 1
        elif len(matched_words) > 0:
            # If we've started matching but this word doesn't match, reset
            matched_words = []
            i = 0
    return matched_words if len(matched_words) == len(quote_words) else []


def get_bounding_box(words):
    if not words:
        return None
    x_min = min(min(word["polygon"][0::2]) for word in words)
    y_min = min(min(word["polygon"][1::2]) for word in words)
    x_max = max(max(word["polygon"][0::2]) for word in words)
    y_max = max(max(word["polygon"][1::2]) for word in words)
    return [x_min, y_min, x_max, y_max]


def highlight_quote(image_path, quote, page_words):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    matched_words = find_sentence_words(quote, page_words)
    if matched_words:
        bounding_box = get_bounding_box(matched_words)
        if bounding_box:
            draw.rectangle(bounding_box, outline="red", width=2)
    else:
        print(f"Could not find a match for: {quote}")

    return img


def process_quotes(model_answer, ocr_results, image_paths):
    for quote_info in model_answer["quotes"]:
        quote = quote_info["quote"]
        page = quote_info["page"] - 1  # Adjust for 0-based indexing

        if page < len(ocr_results):
            page_words = ocr_results[page]["pages"][0]["words"]
            highlighted_img = highlight_quote(image_paths[page], quote, page_words)

            # Save the highlighted image
            output_path = f"output/highlighted_page_{page+1}.jpg"
            highlighted_img.save(output_path)
            print(f"Highlighted image saved as {output_path}")
        else:
            print(f"Page {page+1} not found in OCR results")
