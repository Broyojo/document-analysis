import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Flag to enable/disable quadratic fit
include_quadratic_fit = False

# Load data from the JSON file
with open("eval_data_ocr_and_image_400.json", "r") as file:
    try:
        data = json.load(file)["data"]
    except:
        data = json.load(file)


# Extract number of documents (pages) and response times
num_docs = np.array([len(item["page_ids"]) for item in data])
response_times = np.array([item["response_time"] for item in data])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(num_docs, response_times)

# Create the plot
plt.figure(figsize=(12, 7))
plt.scatter(num_docs, response_times, color="blue", alpha=0.7)

# Add linear regression line
x_range = np.linspace(min(num_docs), max(num_docs), 100)
plt.plot(
    x_range,
    slope * x_range + intercept,
    color="red",
    linestyle="--",
    label=f"Linear: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}",
)

# Add quadratic fit if enabled
if include_quadratic_fit:
    # Perform quadratic fit
    coeffs = np.polyfit(num_docs, response_times, 2)
    poly = np.poly1d(coeffs)

    # Calculate R-squared for quadratic fit
    yhat = poly(num_docs)
    ybar = np.sum(response_times) / len(response_times)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((response_times - ybar) ** 2)
    r_squared = ssreg / sstot

    # Plot quadratic fit
    plt.plot(
        x_range,
        poly(x_range),
        color="green",
        linestyle=":",
        label=f"Quadratic: y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}\nR² = {r_squared:.4f}",
    )

# Customize the plot
plt.title("Response Time vs Number of Pages")
plt.xlabel("Number of Pages")
plt.ylabel("Response Time (seconds)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Optional: Save the plot as an image file
# plt.savefig('response_time_vs_num_docs.png')
