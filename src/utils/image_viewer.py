import numpy as np
from matplotlib import pyplot as plt
import PIL as pil

def show_images(
        images:list[np.ndarray],
        columns:int,
        rows:int,
        labels:list[str]=None):
    """shox images or graphs in a grid

    Parameters
    ----------
    images : list[np.ndarray]
        images or graphs to show
    rows : int
        number of images/ graphs per a row
    columns : int
        how many rows
    labels : list[str], by default None
        labels per image, optional
    """
    n = len(images)
    fig, axes = plt.subplots(rows, columns, figsize=(10, 10))

    # plot images on axes grid
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            # check if is graph or image
            if isinstance(images[i], np.ndarray) and images[i].ndim == 1:
                ax.plot(images[i])
                ax.axis('on')
            else:
                ax.imshow(images[i])
                ax.axis('off')
            if labels:
                try:
                    ax.set_title(labels[i])
                except IndexError:
                    pass
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def overlay_images(image_1:np.ndarray, image_2:np.ndarray,opacity:float) -> np.ndarray:
    """overlays two images with a given opacity (0-1), op

    Parameters
    ----------
    image_1 : np.ndarray
        first image
    image_2 : np.ndarray
        second image
    opactiy : float
        opacity

    Returns
    -------
    np.ndarray
        blended image
    """
    assert image_1.shape == image_2.shape, "Images must have the same shape"

    # Ensure opacity is within the valid range [0, 1]
    opacity = max(0, min(1, opacity))


    overlay = opacity * image_1 + (1 - opacity) * image_2

    return np.clip(overlay, 0, 255).astype(np.uint8)

if "__main__" == __name__:
    images_1 = np.array(pil.Image.open(r"D:\Users\Desktop\Uni\Master\Neural Networks (NN)\Seminar\reliability_metrics_to_evaluate_saliency_maps\data\CUB_200_2011\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg"))
    images_2 = np.array(pil.Image.open(r"D:\Users\Desktop\Uni\Master\Neural Networks (NN)\Seminar\reliability_metrics_to_evaluate_saliency_maps\data\CUB_200_2011\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0002_55.jpg"))
    plt.imshow(pil.Image.blend(images_1, images_1, 0.5))
    labels = ["Bird", "Saliency","Bird2", "Saliency2"]
    images = [
        np.array(pil.Image.open(r"D:\Users\Desktop\Uni\Master\Neural Networks (NN)\Seminar\reliability_metrics_to_evaluate_saliency_maps\data\CUB_200_2011\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg")),
        np.sin(np.linspace(0, 2*np.pi, 100)),
        pil.Image.open(r"D:\Users\Desktop\Uni\Master\Neural Networks (NN)\Seminar\reliability_metrics_to_evaluate_saliency_maps\data\CUB_200_2011\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg"),
        pil.Image.open(r"D:\Users\Desktop\Uni\Master\Neural Networks (NN)\Seminar\reliability_metrics_to_evaluate_saliency_maps\data\CUB_200_2011\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg"),
        pil.Image.open(r"D:\Users\Desktop\Uni\Master\Neural Networks (NN)\Seminar\reliability_metrics_to_evaluate_saliency_maps\data\CUB_200_2011\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg"),
        np.cos(np.linspace(0, 2*np.pi, 100))]
    show_images(images,2,4,labels)

