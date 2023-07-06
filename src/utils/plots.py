import itertools
from pathlib import Path
import pickle
import traceback
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_color_palette(num_colors):
    palette = []

    # Generate a random hue value for the first color
    hue = random.randint(0, 360)
    palette.append(hue_to_rgb(hue))

    # Generate the remaining colors
    for _ in range(num_colors - 1):
        # Randomly select a hue that is different from the previous hue
        hue = (hue + random.randint(30, 90)) % 360
        palette.append(hue_to_rgb(hue))

    return palette

def hue_to_rgb(hue):
    # Convert HSL color to RGB color
    saturation = random.uniform(0.4, 0.6)
    lightness = random.uniform(0.4, 0.6)

    c = (1 - abs(2 * lightness - 1)) * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = lightness - c / 2

    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    elif 240 <= hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = float((r + m) )
    g = float((g + m) )
    b = float((b + m) )

    return r, g, b


def load_data(input_dir: str, extension: str = "pkl") -> dict:
    result = {}
    for path in Path(input_dir).rglob(f"*.{extension}"):
        method = path.parent.name
        metric = path.stem
        try:
            print(f"Loading {method} {metric} from {path}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            try:
                image_idx, scores = zip(*data)
                # Remove nan values
                # scores = [x for x in scores if str(x) != "nan"]
                scores = np.array(list(scores), dtype=np.float64).flatten()
            except:
                image_idx, data_dicts = zip(*data)
                scores = np.array([list(dict_.values())[0]
                                   for dict_ in data_dicts], dtype=np.float64)
            result.update({
                (method, metric): (image_idx, scores)
            })
        except Exception as e:
            print(f"Failed to load {method} {metric} from {path}")
            print(traceback.format_exc())
    return result


def load_mi(input_dir: Path):
    print(f"Loading {input_dir} ")
    result = {}
    # load data
    with open(input_dir, 'rb') as f:
        data = pickle.load(f)

    for dict_ in data:
        dict_values = list(dict_.values())
        attribution = dict_values[1]

        if not attribution in result:
            result.update(
                {attribution: {
                    "metrics": [],
                    "mi": [],
                    "mig_x": [],
                    "mig_y": []}
                 })
        metric = dict_values[0]
        attribution = dict_values[1]
        # append values to lists
        result[attribution]["metrics"].append(metric)
        result[attribution]["mi"].append(dict_values[2])
        result[attribution]["mig_x"].append(dict_values[3])
        result[attribution]["mig_y"].append(dict_values[4])
        attribution = dict_values[1]
    return result


def create_histogram(
        data,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins="auto",
        output_path: str = None,
        show: bool = False,
        color: str = None
):
    if not color:
        color = generate_color_palette(1)[0]

    # Compute the histogram
    _, bins = np.histogram(data, bins=bins)

    # Plot the histogram
    plt.hist(data, bins=bins,color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved histogram to {output_path}")
    plt.close()


def create_bar_chart(
        data: list[np.ndarray],
        title: str = "Bar Chart of Mean Values",
        xlabel: str = "Data",
        ylabel: str = "Mean values",
        tick_labels: list[str] = ['Data 1', 'Data 2', 'Data 3'],
        output_path: str = "output/bar_chart.png",
        show: bool = False,
        color: str = None
):
    try:
        if not color:
            color = generate_color_palette(1) * len(data)
        fig, ax = plt.subplots()
        # Create the bar chart
        position = range(len(data))
        bars = ax.bar(position, data, align='center',color=color)
        ax.bar_label(bars, fmt='%.2f')
        tick_labels = [f"({label[0]} , {label[1]})" for label in tick_labels]
        ax.set_xticks(position, tick_labels, rotation='vertical')
        ax.set_ylim(0, max(data) + 0.2)
        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # Customize x-axis tick labels
        if show:
            plt.show()
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    except:
        print("Error in create_bar_chart")
        print(traceback.format_exc())


def create_scatter(x: list[float], y: list[float, ], title: str = "Scatter Plot", x_label: str = "x", y_label: str = "y", output_path: str = None):
    try:
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
    except:
        pass

def create_bar(data: dict, sort: bool = False):
    for attribution, values in data.items():
        if sort:
            # create save paths
            save_path_mi = Path(
                "output", "bar_sorted", f"{attribution}_mi.png")
            save_path_mig_x = Path(
                "output", "bar_sorted", f"{attribution}_mig_x.png")
            save_path_mig_y = Path(
                "output", "bar_sorted", f"{attribution}_mig_y.png")
        else:
            save_path_mi = Path(
                "output", "bar", f"{attribution}_mi.png")

            save_path_mig_x = Path(
                "output", "bar", f"{attribution}_mig_x.png")
            save_path_mig_y = Path(
                "output", "bar", f"{attribution}_mig_y.png")
        save_path_mi.parent.mkdir(parents=True, exist_ok=True)
        save_path_mig_x.parent.mkdir(parents=True, exist_ok=True)
        save_path_mig_y.parent.mkdir(parents=True, exist_ok=True)
        # sort lists by mi
        if sort:
            mi, x_labels = zip(
                *sorted(zip(data[attribution]["mi"], data[attribution]["metrics"])))
        else:
            mi = data[attribution]["mi"]
            x_labels = data[attribution]["metrics"]
        # create bar chart for mi
        create_bar_chart(mi, title=f"{attribution} - Mutual Information", xlabel="Metrics",
                         ylabel=f"Mutual Information", tick_labels=x_labels, output_path=save_path_mi)

        if sort:
            mig_x, x_labels = zip(
                *sorted(zip(data[attribution]["mig_x"], data[attribution]["metrics"])))
        else:
            mig_x = data[attribution]["mig_x"]
            x_labels = data[attribution]["metrics"]
        # create bar chart for mig_x
        create_bar_chart(mig_x, title=f"{attribution} - Mutual Information Gap for x", xlabel="Metrics",
                         ylabel=f"Mutual Information", tick_labels=x_labels, output_path=save_path_mig_x)

        if sort:
            # sort lists by mig_y
            mig_y, x_labels = zip(
                *sorted(zip(data[attribution]["mig_y"], data[attribution]["metrics"])))
        else:
            mig_y = data[attribution]["mig_y"]
            x_labels = data[attribution]["metrics"]
        # create bar chart for mig_y
        create_bar_chart(mig_y, title=f"{attribution} - Mutual Information Gap for y", xlabel="Metrics",
                         ylabel=f"Mutual Information", tick_labels=x_labels, output_path=save_path_mig_y)


def create_mean_bar(data: dict, sort: bool = False):
    # get mean mi for each metric combination
    mean_dict = {}
    save_path_mi_mean = Path("output", f"mean_mi.png")
    if sort:
        save_path_mi_mean = Path("output", f"mean_mi_sorted.png")
    save_path_mi_mean.parent.mkdir(parents=True, exist_ok=True)
    for attribution, values in data.items():
        for metric in values["metrics"]:
            if not metric in mean_dict:
                mean_dict.update({metric: []})
            mean_dict[metric].append(
                values["mi"][values["metrics"].index(metric)])
    # calculate mean mi for each metric combination
    mean_dict = {key: np.mean(value) for key, value in mean_dict.items()}
    x_labels = mean_dict.keys()
    data = mean_dict.values()
    if sort:
        data, x_labels = zip(
            *sorted(zip(data, x_labels)))
    create_bar_chart(data, title=f"Mean Mutual Information", xlabel="Metrics",
                     ylabel=f"Mutual Information", tick_labels=x_labels, output_path=save_path_mi_mean)


def create_histograms(data: dict, bins: int = 10):
    for metric, values in data.items():
        save_path_hist = Path(
            "output", "histograms",f"{metric[0]}" ,f"{metric[1]}_histogramm_{bins}.png")
        save_path_hist.parent.mkdir(parents=True, exist_ok=True)
        create_histogram(
            values[-1], title=f"{metric[0]} - {metric[1]}", xlabel=f"{metric[1]} scores", ylabel="absolute Frequency", output_path=save_path_hist, bins=bins)


def create_scatter_plot(data: dict):
    all_combis = list(itertools.combinations(data.keys(), 2))
    for combi in all_combis:
        attribution = combi[0][0]
        metric_1 = combi[0][1]
        metric_2 = combi[1][1]
        output_path = Path("output", "scatter_plots",attribution,
                           f"{attribution}_{metric_1}_{metric_2}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_scatter(data[combi[0]][1], data[combi[1]][1],
                       title=f"{attribution}- {metric_1} {metric_2}", x_label=metric_1, y_label=metric_2, output_path=output_path)


if __name__ == "__main__":
    in_dir = r"output"
    data = load_mi(Path(r"data/mi.pkl"))
    create_bar(data, sort=False)
    create_bar(data, sort=True)
    create_mean_bar(data, sort=False)
    create_mean_bar(data, sort=True)
    data = load_data(in_dir)
    create_scatter_plot(data)
    create_histograms(data)
    create_histograms(data,"auto")
