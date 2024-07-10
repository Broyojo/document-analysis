import math


def wilson_score_interval(p, n, z=1.96):
    """
    Calculate Wilson score interval

    :param p: observed proportion (accuracy)
    :param n: total number of samples
    :param z: z-score (1.96 for 95% confidence interval)
    :return: lower and upper bounds of the confidence interval
    """
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    lower_bound = (
        centre_adjusted_probability - adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + adjusted_standard_deviation
    ) / denominator

    return (lower_bound, upper_bound)


def calculate_sample_size(p, margin_of_error, confidence_level=0.95):
    """
    Calculate the required sample size for a desired margin of error

    :param p: expected proportion (can use observed proportion as an estimate)
    :param margin_of_error: desired margin of error (as a proportion)
    :param confidence_level: desired confidence level (default 0.95 for 95% confidence)
    :return: required sample size
    """
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence_level]
    numerator = z**2 * p * (1 - p)
    denominator = margin_of_error**2
    return math.ceil(numerator / denominator)


def find_sample_size_for_width(
    initial_p,
    target_width,
    confidence_level=0.95,
    start=100,
    step=100,
    max_iterations=1000,
):
    """
    Find the sample size that achieves a specific confidence interval width

    :param initial_p: initial proportion estimate
    :param target_width: desired width of the confidence interval
    :param confidence_level: desired confidence level (default 0.95 for 95% confidence)
    :param start: starting sample size for the search
    :param step: step size for increasing the sample size
    :param max_iterations: maximum number of iterations to prevent infinite loop
    :return: required sample size
    """
    n = start
    for _ in range(max_iterations):
        lower, upper = wilson_score_interval(
            initial_p, n, z={0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence_level]
        )
        current_width = upper - lower
        if current_width <= target_width:
            return n
        n += step
    raise ValueError(
        "Could not find appropriate sample size within the given number of iterations"
    )


# Your initial data
initial_accuracy = 0.6862745098039215

# Calculate for ±5% margin of error
sample_size_5percent = find_sample_size_for_width(
    initial_accuracy, 0.10
)  # 0.10 because ±5% is a total width of 10%

# Calculate for ±1% margin of error
sample_size_1percent = find_sample_size_for_width(
    initial_accuracy, 0.02
)  # 0.02 because ±1% is a total width of 2%

print(f"For ±5% margin of error: Approximately {sample_size_5percent} samples needed")
print(f"For ±1% margin of error: Approximately {sample_size_1percent} samples needed")

# Verify the results
for n in [sample_size_5percent, sample_size_1percent]:
    lower, upper = wilson_score_interval(initial_accuracy, n)
    print(f"\nFor {n} samples:")
    print(f"95% Confidence Interval: ({lower:.4f}, {upper:.4f})")
    print(f"Interval width: {upper - lower:.4f}")
