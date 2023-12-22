import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#####

def extract_eval_data(eval_csv):
    ''' This Function extracts the eval data from eval_csv
    Args:
        eval_csv (os.PathLike): eval csv file path
    Returns:
        img_paths (list): list of [label, image_path]
    '''
    df = pd.read_csv(eval_csv)
    return [os.path.join( "/eval", "images", x) for x in df['ImageID']]
def show_test_images(test_data: list, rows: int, cols: int):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    random_idxs = np.random.randint(len(test_data), size=rows*cols)
    axes = axes.flatten()
    for i, idx in enumerate(random_idxs):
        image = Image.open(test_data[idx]).convert('RGB')
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
test_data = extract_eval_data("/eval/info.csv")
show_test_images(test_data, 8, 11)

####