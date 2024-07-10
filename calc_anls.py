import json
import numpy as np
from anls import anls_score


def process_eval_data(file_path, anls_threshold=0.5):
    with open(file_path, "r") as file:
        data = json.load(file)

    anls_scores = []
    for item in data:
        model_answer = item["model_answer"]
        ground_truths = item["answers"]
        score = anls_score(
            prediction=model_answer, gold_labels=ground_truths, threshold=anls_threshold
        )
        anls_scores.append(score)

    overall_anls = np.mean(anls_scores)
    return overall_anls, anls_scores


def main():
    file_path = "eval_data50.json"
    anls_threshold = 0.5  # You can adjust this if needed

    overall_anls, individual_scores = process_eval_data(file_path, anls_threshold)

    print(f"Overall ANLS score: {overall_anls:.4f}")
    print(f"Number of questions evaluated: {len(individual_scores)}")
    print(f"Individual ANLS scores: {individual_scores}")


if __name__ == "__main__":
    main()
