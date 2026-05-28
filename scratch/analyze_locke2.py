import os
from PIL import Image
import numpy as np

SRC = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779875861495.png"

def main():
    img = Image.open(SRC)
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    # Scan for non-green columns within the IDLE row (y=52..170)
    idle_slice = arr[52:170, :, :]
    
    print("--- Column green fraction in IDLE row ---")
    col_greens = []
    for x in range(w):
        col = idle_slice[:, x, :]
        r, g, b = col[:,0].astype(float), col[:,1].astype(float), col[:,2].astype(float)
        green_pct = ((g > 120) & (g - r > 50) & (g - b > 50)).mean()
        col_greens.append(green_pct)
    
    # Find separator columns (fully or nearly green, surrounded by content)
    print("\nDark/separator column groups (green_pct < 0.3):")
    in_sep = False
    start = 0
    seps = []
    for x, gp in enumerate(col_greens):
        # Dark separators have low green (black bars) — but we saw none
        # Instead look for the column borders by checking darkness
        pass
    
    # Directly check average darkness of columns
    print("\nColumn darkness scan:")
    for x in range(w):
        col = idle_slice[:, x, :]
        darkness = col.mean()
        if darkness < 30:
            print(f"  Dark col x={x} (mean={darkness:.1f})")
    
    # Save the entire IDLE row as a single image to visually inspect
    idle_img = Image.fromarray(arr[52:170, :, :])
    idle_img.save("scratch/locke_idle_row.png")
    print("\nSaved IDLE row to scratch/locke_idle_row.png")
    
    # Also crop with 130px cells to see if it matches
    print("\n--- Trying 130px cell width ---")
    for col in range(8):
        x1 = col * 130
        x2 = min((col+1)*130, w)
        cell = arr[52:170, x1:x2, :]
        r, g, b = cell[:,:,0].astype(float), cell[:,:,1].astype(float), cell[:,:,2].astype(float)
        non_green = ~((g > 120) & (g - r > 50) & (g - b > 50))
        content_pct = non_green.mean()
        print(f"  Col {col} (x={x1}..{x2}): content={content_pct*100:.1f}%")

if __name__ == "__main__":
    main()
