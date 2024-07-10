import base64
import inspect
import json
import random
import time
from openai import OpenAI
from tqdm import tqdm
from calc_ci import wilson_score_interval

SEED = 69420

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


def load_dataset(path, sample_ratio=0.01, k=None):
    with open(path) as f:
        data = json.load(f)["data"]
        print(len(data))
    return random.sample(data, k=int(sample_ratio * len(data)) if k is None else k)


# TODO: need about 400 samples for 95% confidence interval with 5% margin of error
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
                "content": "You are an assistant which extracts information from documents. Output the final answer with no other extraneous text.",  # TODO: experiment with better prompt structure (https://arxiv.org/pdf/2312.16171)
            },
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
            },
        ]

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
                "response_time": response_time,
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
finally:
    print("Accuracy:", total_correct / count * 100)
    with open("eval_data.json", "w") as f:
        json.dump(eval_data, f, indent=4)
