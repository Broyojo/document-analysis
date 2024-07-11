import base64
import inspect
import json
import logging
import random
import sys
import time
from openai import OpenAI
from tqdm import tqdm
from calc_ci import wilson_score_interval
from ocr import ocr, parallel_ocr
import traceback

SEED = 12345

# currently doing: image first, ocr second in prompt
ENABLE_OCR = True
ENABLE_IMAGE = True

random.seed(SEED)

# very evil hack
old_print = print


def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.write(*args, **kwargs)
    except:
        old_print(*args, **kwargs)


# globaly replace print with new_print
inspect.builtins.print = new_print

# set up some logging (TODO: when this project is more robust make a good logger and unit testing, this script is too fragile)
# TODO: also add some kind of token calculation thing so that I can accurately measure the number of tokens. i'm just going by dollars spent now through the billing

# # Set up logging to write to a file
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     filename="output.log",
#     filemode="a",  # 'a' means append (add to the end of the file)
# )

# # Create a custom logger
# logger = logging.getLogger()

# # Create handlers
# c_handler = logging.StreamHandler()  # Console handler
# f_handler = logging.FileHandler("output.log")  # File handler

# # Set level for handlers
# c_handler.setLevel(logging.INFO)
# f_handler.setLevel(logging.INFO)

# # Add handlers to the logger
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)


# # Define a custom print function
# def custom_print(*args, **kwargs):
#     print(*args, **kwargs)  # This will print to stdout
#     # Log the printed message
#     logger.info(" ".join(map(str, args)))


# # Replace the built-in print function with our custom function
# sys.stdout.write = custom_print


def load_dataset(path, sample_ratio=0.01, k=None):
    with open(path) as f:
        data = json.load(f)["data"]
        print(len(data))
    return random.sample(data, k=int(sample_ratio * len(data)) if k is None else k)


# note: need about 400 samples for 95% confidence interval with 5% margin of error
dataset = load_dataset("./dataset/mp-docvqa/val.json", k=400)
print("Loaded dataset with", len(dataset), "samples")


def encode_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


client = OpenAI()

total_correct = 0
count = 0

eval_data = []

try:
    bar = tqdm(dataset)
    for sample in bar:
        question = sample["question"]
        question_id = sample["questionId"]
        doc_id = sample["doc_id"]
        page_ids = sample["page_ids"]
        answers = sample["answers"]
        answer_page_idx = sample["answer_page_idx"]

        prompt = [
            {
                "role": "system",
                "content": "You are an assistant which extracts information from documents. Output the final answer with no other extraneous text.",  # TODO: experiment with better prompt structure (https://arxiv.org/pdf/2312.16171). maybe add <thinking></thinking> tokens? however must be careful to not balloon latency and cost
            }
        ]

        if ENABLE_IMAGE and ENABLE_OCR:
            image_paths = [f"/mnt/ssd/images/{page_id}.jpg" for page_id in page_ids]
            ocr_start = time.time()
            try:
                ocr_results = parallel_ocr(image_paths, markdown=True)
            except:
                print("encountered error during OCR")
                traceback.print_exc()
                continue

            ocr_time = time.time() - ocr_start
            input = [{"type": "text", "text": question}]

            for image_path, ocr_result in zip(image_paths, ocr_results):
                input.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_base64(image_path)}"
                            },
                        },
                        {"type": "text", "text": ocr_result["content"]},
                    ]
                )

            prompt.append({"role": "user", "content": input})
        elif ENABLE_IMAGE:
            prompt.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_base64(image_path)}"
                            },
                        }
                        for image_path in map(
                            lambda page_id: f"/mnt/ssd/images/{page_id}.jpg",
                            page_ids,
                        )
                    ],
                }
            )
        elif ENABLE_OCR:
            ocr_concated = ""
            ocr_start = time.time()
            image_paths = [f"/mnt/ssd/images/{page_id}.jpg" for page_id in page_ids]

            try:
                ocr_results = parallel_ocr(image_paths, markdown=True)
            except:
                print("encountered error during OCR")
                traceback.print_exc()
                continue

            for result in ocr_results:
                ocr_concated += content + "\n\n"

            prompt.append(
                {"role": "user", "content": question + "\n\n\n" + ocr_concated}
            )
            ocr_time = time.time() - ocr_start

        print("=================== QUESTION ===================")

        print(question)
        print("Pages:")
        for i, page_id in enumerate(page_ids):
            print(f"{i+1}. /mnt/ssd/images/{page_id}.jpg")

        print(f"=================== GROUND TRUTH ===================")
        print(answers)

        print("=================== MODEL ANSWER ===================")
        start = time.time()
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
        response_time = time.time() - start

        for answer in answers:
            if answer.lower() in model_answer.lower():
                print("Correct!")
                total_correct += 1
                break
        else:
            print("Incorrect!")

        eval_data.append(
            {
                "question": question,
                "question_id": question_id,
                "doc_id": doc_id,
                "page_ids": page_ids,
                "answers": answers,
                "answer_page_idx": answer_page_idx,
                "model_answer": model_answer,
                "response_time": response_time + (ocr_time if ENABLE_OCR else 0),
            }
        )

        count += 1
        accuracy = total_correct / count
        lower, upper = wilson_score_interval(accuracy, count)
        dist = (upper - lower) / 2
        bar.set_description(
            f"Accuracy: {total_correct / count * 100:.2f}% ± {dist * 100:.2f}%",
            refresh=True,
        )
except KeyboardInterrupt:
    print("Interrupted")
except Exception as e:
    print(traceback.format_exc())
finally:
    accuracy = total_correct / count
    lower, upper = wilson_score_interval(accuracy, count)
    dist = (upper - lower) / 2
    acc_str = f"{total_correct / count * 100:.2f}% ± {dist * 100:.2f}%"
    print(
        f"Final Accuracy: {acc_str}",
    )
    with open("eval_data.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "enable_ocr": ENABLE_OCR,
                "enable_image": ENABLE_IMAGE,
                "accuracy": accuracy,
                "accuracy_stdev": dist,
                "data": eval_data,
            },
            f,
            indent=4,
        )
