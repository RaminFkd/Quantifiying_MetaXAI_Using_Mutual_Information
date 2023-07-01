from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_data(input_dir: str, extension: str = "pkl") -> dict:
    result = {}
    for path in Path(input_dir).rglob(f"*.{extension}"):
        method = path.parent.name
        metric = path.stem
        print(f"Loading {method} {metric} from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        try:
            image_idx, scores = zip(*data)
            scores = np.array(list(scores), dtype=np.float64).flatten()
        except TypeError:
            image_idx, data_dicts = zip(*data)
            scores = np.array([list(dict_.values())[0]
                              for dict_ in data_dicts], dtype=np.float64)

        result.update({
            (method, metric): (image_idx, scores)
        })
    return result


def create_histogram(
        data,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins="auto"
):
    # Create a random array of data

    # Compute the histogram
    hist, bins = np.histogram(data, bins=bins)

    # Plot the histogram
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def create_bar_chart(
        data: list[np.ndarray],
        title: str = "Bar Chart of Mean Values",
        xlabel: str = "Data",
        ylabel: str = "Mean values",
        tick_labels: list[str] = ['Data 1', 'Data 2', 'Data 3']
):
    # Calculate the means
    means = [np.mean(arr) for arr in data]

    # Generate x-axis values
    x = np.arange(len(means))

    # Create the bar chart
    plt.bar(x, means)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Customize x-axis tick labels
    plt.xticks(x, tick_labels)
    plt.show()


if __name__ == "__main__":
    loaded_data = load_data("output")
    for metric, values in loaded_data.items():
        create_histogram(
            values[-1], title=f"{metric[0]} {metric[1]}", xlabel="DAUC", ylabel="Frequency")
    bar_values = [values[-1] for values in loaded_data.values()]
    create_bar_chart(bar_values, title=f"Mean values", xlabel="DAUC", ylabel="Mean DAUC", tick_labels=[
                     f"{metric[0]} {metric[1]}" for metric in loaded_data.keys()])
    pass
