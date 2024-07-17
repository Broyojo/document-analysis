import tempfile
from ocr import ocr
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
def pipe(doc_path: str, queries: list[str]):
    image_paths, temp_dir = save_pdf_to_images(doc_path)
    search_results = search_top_k(image_paths, queries, k=5)[0]

    prompt = [
        {
            "role": "system",
            "content": "You are an assistant which extracts information from documents. Output the final answer with no other extraneous text. With your answer, please provide the direct quotes from the document that support your answer. Here is the output schema: {'answer': '...', 'quotes': [{'quote': '...', 'page': 42}, ...]}.",
        }
    ]

    input = [{"type": "text", "text": queries[0]}]

    for result in search_results:
        input.extend(
            [
                {
                    "type": "text",
                    "text": f"Page {image_paths.index(result.doc_path)}",  # true page number
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
        messages=prompt,
        stream=True,
    ):
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
            model_answer += content
    print()


if __name__ == "__main__":
    doc_path = "/home/broyojo/Downloads/2212.05935v2.pdf"
    pipe(doc_path, ["What is the system diagram showing?"])
