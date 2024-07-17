from collections import defaultdict
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
                "prebuilt-layout",
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


def find_best_sentence_match(quote, page_words, threshold=50):
    quote_words = quote.lower().split()
    best_match = []
    best_score = 0

    for i in range(len(page_words) - len(quote_words) + 1):
        window = page_words[i : i + len(quote_words)]
        score = sum(
            fuzz.ratio(q, w["content"].lower()) for q, w in zip(quote_words, window)
        ) / len(quote_words)

        if score > best_score:
            best_score = score
            best_match = window

    return best_match if best_score >= threshold else []


def get_bounding_box(words):
    if not words:
        return None
    x_min = min(min(word["polygon"][0::2]) for word in words)
    y_min = min(min(word["polygon"][1::2]) for word in words)
    x_max = max(max(word["polygon"][0::2]) for word in words)
    y_max = max(max(word["polygon"][1::2]) for word in words)
    return [x_min, y_min, x_max, y_max]


def highlight_quotes(image_path, quotes, page_words):
    img = Image.open(image_path).convert("RGBA")
    highlight_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(highlight_layer)

    for quote in quotes:
        matched_words = find_best_sentence_match(quote, page_words)
        if matched_words:
            bounding_box = get_bounding_box(matched_words)
            if bounding_box:
                # Draw a semi-transparent yellow highlight
                draw.rectangle(bounding_box, fill=(255, 255, 0, 80))
            print(f"✅ Match found for: {quote}")
            print(f"=> Matched words: {[word['content'] for word in matched_words]}")
        else:
            print(f"❌ Could not find a match for: {quote}")

    # Combine the original image with the highlight layer
    highlighted_img = Image.alpha_composite(img, highlight_layer).convert("RGB")
    return highlighted_img


def process_quotes(model_answer, ocr_results, image_paths):
    # Group quotes by page
    quotes_by_page = defaultdict(list)
    for quote_info in model_answer["quotes"]:
        quotes_by_page[quote_info["page"]].append(quote_info["quote"])

    for page, quotes in quotes_by_page.items():
        page_index = page - 1  # Adjust for 0-based indexing
        if page_index < len(ocr_results):
            page_words = ocr_results[page_index]["pages"][0]["words"]
            highlighted_img = highlight_quotes(
                image_paths[page_index], quotes, page_words
            )

            # Save the highlighted image
            output_path = f"output/highlighted_page_{page}.jpg"
            highlighted_img.save(output_path)
            print(f"Highlighted image saved as {output_path}")
        else:
            print(f"Page {page} not found in OCR results")
