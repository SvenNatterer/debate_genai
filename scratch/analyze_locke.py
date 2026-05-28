import os
from PIL import Image
import numpy as np

SRC = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779875861495.png"

def main():
    img = Image.open(SRC)
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image size: {w}x{h}")

    # Find black bar rows (mean brightness < 15)
    print("\n--- Black rows (separators) ---")
    black_rows = []
    for y in range(h):
        mean = arr[y].mean()
        if mean < 15:
            black_rows.append(y)

    if black_rows:
        # Group consecutive
        groups = []
        start = black_rows[0]
        prev = black_rows[0]
        for y in black_rows[1:]:
            if y > prev + 1:
                groups.append((start, prev))
                start = y
            prev = y
        groups.append((start, prev))
        for g in groups:
            print(f"  Black bar: y={g[0]}..{g[1]} (height={g[1]-g[0]+1})")
    else:
        print("  No black rows found")

    # Find black bar cols (mean brightness < 15)
    print("\n--- Black columns (cell separators) ---")
    black_cols = []
    for x in range(w):
        mean = arr[:, x].mean()
        if mean < 15:
            black_cols.append(x)
    if black_cols:
        groups = []
        start = black_cols[0]
        prev = black_cols[0]
        for x in black_cols[1:]:
            if x > prev + 1:
                groups.append((start, prev))
                start = x
            prev = x
        groups.append((start, prev))
        for g in groups:
            print(f"  Black col: x={g[0]}..{g[1]}")
    else:
        print("  No black cols found")

    # Also scan for the label bars (dark text band below each row)
    print("\n--- Rows with low green fraction (likely label/text rows) ---")
    for y in range(h):
        row = arr[y]
        r, g, b = row[:,0].astype(float), row[:,1].astype(float), row[:,2].astype(float)
        green_pixels = ((g > 120) & (g - r > 50) & (g - b > 50)).sum()
        if green_pixels < w * 0.3 and arr[y].mean() > 10:
            print(f"  y={y}: non-green row (green%={100*green_pixels/w:.0f}%, mean={arr[y].mean():.1f})")

if __name__ == "__main__":
    main()
