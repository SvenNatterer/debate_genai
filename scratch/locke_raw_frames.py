import os
from PIL import Image
import numpy as np

SRC = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779875861495.png"

def main():
    img = Image.open(SRC)
    arr = np.array(img)

    # Save individual frames 3 and 4 scaled 4x to see what's at the edges
    for col in range(6):
        x1 = col * 130
        x2 = (col + 1) * 130
        cell = arr[52:170, x1:x2, :]
        
        # What's in the leftmost and rightmost columns of each cell?
        left5  = cell[:, :5, :]
        right5 = cell[:, -5:, :]
        
        # Print mean RGB of left/right edges
        print(f"Col {col}: left5_mean={left5.mean():.1f}  right5_mean={right5.mean():.1f}")
        
        # Save upscaled frame
        frame_img = Image.fromarray(cell.astype(np.uint8))
        scaled = frame_img.resize((frame_img.width * 4, frame_img.height * 4), Image.Resampling.NEAREST)
        scaled.save(f"scratch/locke_raw_frame_{col}.png")
        print(f"  → saved locke_raw_frame_{col}.png")

if __name__ == "__main__":
    main()
