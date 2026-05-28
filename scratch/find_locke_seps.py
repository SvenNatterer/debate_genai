import os
from PIL import Image
import numpy as np

SRC = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779875861495.png"

def main():
    img = Image.open(SRC)
    arr = np.array(img)
    
    # Scan columns in the idle row for dark (border) columns
    idle = arr[52:170, :, :]
    print("Dark columns in IDLE row (mean < 50):")
    for x in range(800):
        col_mean = idle[:, x].mean()
        if col_mean < 50:
            print(f"  x={x} mean={col_mean:.1f}")

if __name__ == "__main__":
    main()
