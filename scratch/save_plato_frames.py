import os
from PIL import Image
import numpy as np

def extract_and_clean_cells(img, start_y, num_cols=6):
    cells = []
    for col in range(num_cols):
        x1 = col * 130
        x2 = (col + 1) * 130
        y1 = start_y
        y2 = start_y + 112
        
        cell = img.crop((x1, y1, x2, y2)).convert("RGBA")
        arr = np.array(cell)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        
        g_f = g.astype(float)
        green_mask = (g > 120) & (g_f - r > 50) & (g_f - b > 50)
        
        border_mask = np.zeros_like(green_mask)
        border_mask[0:2, :] = True
        border_mask[-2:, :] = True
        border_mask[:, 0:10] = True
        border_mask[:, -10:] = True
        
        remove_mask = green_mask | border_mask
        arr[remove_mask, 3] = 0
        
        cleaned_cell = Image.fromarray(arr)
        cells.append(cleaned_cell)
    return cells

def main():
    brain_dir = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012"
    src_path = os.path.join(brain_dir, "media__1779870752198.png")
    
    out_dir = "scratch/plato_frames"
    os.makedirs(out_dir, exist_ok=True)
    
    img = Image.open(src_path)
    frames = extract_and_clean_cells(img, 53, num_cols=6)
    
    for i, f in enumerate(frames):
        out_path = f"{out_dir}/frame_{i}.png"
        # Scale up 3x for easy viewing
        w, h = f.size
        scaled = f.resize((w*3, h*3), Image.Resampling.NEAREST)
        scaled.save(out_path)
        print(f"Saved frame {i} → {out_path}")

if __name__ == "__main__":
    main()
