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


if __name__ == "__main__":
    # Your data
    accuracy = 0.6862745098039215
    total_samples = 51

    lower, upper = wilson_score_interval(accuracy, total_samples)

    print(f"95% Confidence Interval: ({lower:.4f}, {upper:.4f})")
    print(f"This can be interpreted as: {lower:.2%} to {upper:.2%}")
    print(accuracy - lower, upper - accuracy)
