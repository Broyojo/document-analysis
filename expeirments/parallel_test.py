import concurrent.futures


def parallel_ocr(image_paths: list[str], max_workers: int = 5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda x: (x[0], x[1].upper()),
                enumerate(image_paths),
            )
        )

    results.sort(key=lambda x: x[0])
    results = [result[1] for result in results]
    return results


def verify_sorting(original_paths: list[str], results: list[str]) -> bool:
    """
    Verify if the results are correctly sorted based on the original image paths.

    :param original_paths: List of original image paths
    :param results: List of processed results
    :return: True if correctly sorted, False otherwise
    """
    if len(original_paths) != len(results):
        return False

    for original, result in zip(original_paths, results):
        if result != original.upper():
            return False

    return True


# Example usage
image_paths = [f"image{i}" for i in range(1000)]
results = parallel_ocr(image_paths)

# Verify sorting
is_sorted_correctly = verify_sorting(image_paths, results)
print(f"Results are sorted correctly: {is_sorted_correctly}")

# Print first 10 results for demonstration
for result in results[:10]:
    print(result)
