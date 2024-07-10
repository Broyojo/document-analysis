import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def compute_accuracy(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    accuracy_data = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in data:
        num_pages = len(item["page_ids"])
        model_answer = item["model_answer"].lower()
        answers = [answer.lower() for answer in item["answers"]]

        is_correct = any(answer in model_answer for answer in answers)

        accuracy_data[num_pages]["total"] += 1
        if is_correct:
            accuracy_data[num_pages]["correct"] += 1

    accuracy_results = {}
    for num_pages, counts in accuracy_data.items():
        accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        accuracy_results[num_pages] = {
            "accuracy": accuracy,
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return accuracy_results


def print_accuracy_results(accuracy_results):
    print("Accuracy Results:")
    print("----------------")
    for num_pages, result in sorted(accuracy_results.items()):
        print(f"Number of Pages: {num_pages}")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Correct: {result['correct']}")
        print(f"  Total: {result['total']}")
        print()


def confidence_interval(correct, total, confidence=0.95):
    if total == 0:
        return 0
    z = stats.norm.ppf((1 + confidence) / 2)
    p = correct / total
    return z * np.sqrt(p * (1 - p) / total)


def plot_accuracy_results(accuracy_results):
    num_pages = list(accuracy_results.keys())
    accuracies = [result["accuracy"] for result in accuracy_results.values()]

    # Calculate confidence intervals
    error_bars = [
        confidence_interval(result["correct"], result["total"])
        for result in accuracy_results.values()
    ]

    plt.figure(figsize=(14, 8))
    bars = plt.bar(
        num_pages,
        accuracies,
        color="skyblue",
        edgecolor="navy",
        yerr=error_bars,
        capsize=5,
    )

    plt.title("Accuracy vs Number of Pages", pad=30)
    plt.xlabel("Number of Pages")
    plt.xticks(np.arange(1, 20, 1))
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
        )

    # Add total count as text below each bar
    for i, (num_page, result) in enumerate(accuracy_results.items()):
        plt.text(i + 1, -0.05, f"n={result['total']}", ha="center", va="top")

    # Fit a line to the data
    x = np.array(num_pages)
    y = np.array(accuracies)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    plt.plot(
        x,
        line,
        color="red",
        linestyle="--",
        label=f"Fit: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    file_path = "eval_data50.json"
    accuracy_results = compute_accuracy(file_path)
    print_accuracy_results(accuracy_results)
    plot_accuracy_results(accuracy_results)
