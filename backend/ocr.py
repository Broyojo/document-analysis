import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import concurrent.futures

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
