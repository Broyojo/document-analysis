import json
import tempfile
from ocr import ocr, process_quotes
from embedding import search_top_k
import base64
from openai import OpenAI
from pdf2image import convert_from_path

client = OpenAI()


def encode_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_pdf_to_images(pdf_path: str) -> list[str]:
    images = convert_from_path(pdf_path)
    image_paths = []

    temp_dir = tempfile.TemporaryDirectory()
    for i, image in enumerate(images):
        with tempfile.NamedTemporaryFile(
            dir=temp_dir.name, suffix=".jpg", delete=False
        ) as tmpfile:
            image_path = tmpfile.name
            image.save(image_path, "jpeg")
            image_paths.append(image_path)

    return image_paths, temp_dir


# TODO: make this multi-query (run gpt multiple times or do multi-query ranking??)
def pipe(doc_path: str, queries: list[str], image_paths=None):
    if image_paths is None:
        image_paths, temp_dir = save_pdf_to_images(doc_path)

    search_results = search_top_k(image_paths, queries, k=10)[0]

    # TODO: improve this prompt, maybe add CoT before answering instead of only allowing JSON output final answer?
    prompt = [
        {
            "role": "system",
            "content": "You are an assistant which extracts information from documents. Output the final JSON answer with no other extraneous text. With your answer, please provide the direct quotes from the document that support your answer. Here is the output schema: {'answer': '...', 'quotes': [{'quote': '...', 'page': 42}, ...]}. Do NOT put any ellipses within a quote, each quote must be an exact match from the document.",
        }
    ]

    input = [{"type": "text", "text": queries[0]}]

    # TODO: maybe give gpt-4o the OCR results along with the images so that it has an easier time quoting?
    # TODO: a better strategy is probably giving the quotes to embedding model and finding area with highest attention, maybe try to highlight patches which are most relevant to the quote. this will allow for highlighting figures, tables, etc.
    for result in search_results:
        input.extend(
            [
                {
                    "type": "text",
                    "text": f"Page {image_paths.index(result.doc_path)+1}",  # true page number
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_base64(result.doc_path)}"
                    },
                },
            ],
        )

    prompt.append({"role": "user", "content": input})

    model_answer = ""
    for chunk in client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},  # enforce json output
        messages=prompt,
        stream=True,
    ):
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
            model_answer += content
    print()

    model_answer = json.loads(model_answer)

    pages = set()
    for quote in model_answer["quotes"]:
        pages.add(quote["page"])

    sorted_pages = sorted(pages)

    ocr_results = ocr([image_paths[page - 1] for page in sorted_pages])

    map = {page: result for page, result in zip(sorted_pages, ocr_results)}

    process_quotes(model_answer, map, image_paths)

    if image_paths is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    single = True
    if single:
        doc_path = "./test_docs/ts_136101v140300p.pdf"
        pipe(
            doc_path,
            ["which LTE bands are suitable for power classs 1 operation?"],
        )
    else:
        import random
        from pathlib import Path

        with open("../datasets/mp-docvqa/val.json", "r") as f:
            data = [d for d in json.load(f)["data"] if len(d["page_ids"]) == 20]

        sample = random.choice(data)

        print(json.dumps(sample, indent=4))

        files = [
            str(path)
            for path in Path("/mnt/ssd/images/").glob(f"{sample['doc_id']}*.jpg")
        ]

        print(f"loaded {len(files)} pages")

        pipe(
            None,
            [sample["question"]],
            image_paths=files,
        )
