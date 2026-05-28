import os
from PIL import Image
import numpy as np

def main():
    brain_dir = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012"
    src_path = os.path.join(brain_dir, "media__1779870752198.png")
    img = Image.open(src_path)
    arr = np.array(img)
    
    # Check each start_y from 35 to 55 and see what y-range the non-green pixels occupy
    # in column 0 (frame 0) vs column 1 (frame 1)
    for test_y in [35, 40, 43, 45, 48, 50, 53]:
        cell0 = img.crop((0, test_y, 130, test_y + 130)).convert("RGBA")
        a0 = np.array(cell0)
        r, g, b, _ = a0[:,:,0], a0[:,:,1], a0[:,:,2], a0[:,:,3]
        g_f = g.astype(float)
        green = (g > 120) & (g_f - r > 50) & (g_f - b > 50)
        non_green = ~green
        rows = np.where(non_green.any(axis=1))[0]
        if len(rows):
            print(f"start_y={test_y}: frame0 non-green rows: {rows[0]}..{rows[-1]} (height={rows[-1]-rows[0]+1})")
        else:
            print(f"start_y={test_y}: frame0 empty")
    
    # Also show what start_y gives for the overall image row boundaries
    # Find top black bar / row separator lines
    print("\n--- Scanning image rows 0..60 for black rows ---")
    for y in range(60):
        row = arr[y]
        mean_brightness = row.mean()
        if mean_brightness < 20:
            print(f"  y={y}: BLACK ROW (mean={mean_brightness:.1f})")
        elif mean_brightness < 60:
            print(f"  y={y}: dark row (mean={mean_brightness:.1f})")

if __name__ == "__main__":
    main()
