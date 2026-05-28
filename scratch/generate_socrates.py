import os
from PIL import Image
import numpy as np

def extract_and_clean_cells(img, start_y):
    cells = []
    for col in range(5):
        x1 = col * 130
        x2 = (col + 1) * 130
        y1 = start_y
        y2 = start_y + 112
        
        cell = img.crop((x1, y1, x2, y2)).convert("RGBA")
        arr = np.array(cell)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        
        # Chroma key for green background
        g_f = g.astype(float)
        green_mask = (g > 120) & (g_f - r > 50) & (g_f - b > 50)
        
        # Remove dark cell borders and padding near the cell margins
        border_mask = np.zeros_like(green_mask)
        border_mask[0:2, :] = True
        border_mask[-2:, :] = True
        border_mask[:, 0:10] = True
        border_mask[:, -10:] = True
        
        remove_mask = green_mask | border_mask
        arr[remove_mask, 3] = 0
        
        # Edge-despill to remove remaining green border fringe
        a = arr[:, :, 3]
        a_pad = np.pad(a, 1, mode="constant", constant_values=0)
        has_transparent_neighbor = (
            (a_pad[2:, 1:-1] == 0) |
            (a_pad[:-2, 1:-1] == 0) |
            (a_pad[1:-1, 2:] == 0) |
            (a_pad[1:-1, :-2] == 0)
        )
        edge_mask = (a > 0) & has_transparent_neighbor
        padded_edge = np.pad(edge_mask, 1, mode="constant", constant_values=False)
        dilated_edge = (
            padded_edge[2:, 1:-1] |
            padded_edge[:-2, 1:-1] |
            padded_edge[1:-1, 2:] |
            padded_edge[1:-1, :-2] |
            edge_mask
        )
        green_edge = dilated_edge & (g > r) & (g > b)
        arr[green_edge, 1] = np.maximum(r[green_edge], b[green_edge])
        
        cleaned_cell = Image.fromarray(arr)
        cells.append(cleaned_cell)
    return cells

def generate_spritesheet(frames, output_path):
    h_target = 148
    y_anchor = 171
    
    sheet_frames = []
    
    for idx in range(5):
        frame = frames[idx]
        
        # Bounding box of character in this frame
        arr = np.array(frame)
        y_idx, x_idx = np.where(arr[:, :, 3] > 0)
        
        if len(y_idx) == 0:
            print(f"Warning: Frame {idx} is empty!")
            sheet_frames.append(Image.new("RGBA", (176, 176), (0, 0, 0, 0)))
            continue
            
        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        
        cropped_char = frame.crop((xmin, ymin, xmax + 1, ymax + 1))
        
        # Scale proportionally to height=148 using NEAREST
        w_scaled = int(round(cropped_char.width * (h_target / cropped_char.height)))
        scaled_char = cropped_char.resize((w_scaled, h_target), Image.Resampling.NEAREST)
        
        # Create transparent canvas
        canvas = Image.new("RGBA", (176, 176), (0, 0, 0, 0))
        
        # Center horizontally, anchor bottom at y_anchor
        x_offset = (176 - w_scaled) // 2
        y_offset = y_anchor - h_target
        
        canvas.paste(scaled_char, (x_offset, y_offset), scaled_char)
        sheet_frames.append(canvas)
        
    # Stitch 5 frames side-by-side to make 880x176 spritesheet
    spritesheet = Image.new("RGBA", (880, 176), (0, 0, 0, 0))
    for idx, f in enumerate(sheet_frames):
        spritesheet.paste(f, (idx * 176, 0))
        
    spritesheet.save(output_path)
    print(f"Saved Socrates spritesheet to {output_path}")

def main():
    spritesheet_path = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779870315939.png"
    if not os.path.exists(spritesheet_path):
        print(f"Error: Socrates spritesheet not found at {spritesheet_path}")
        return
        
    img = Image.open(spritesheet_path)
    
    os.makedirs("images", exist_ok=True)
    
    print("\n--- Processing Socrates ---")
    socrates_frames = extract_and_clean_cells(img, 53)
    generate_spritesheet(socrates_frames, "images/Socrates_idle_strip.png")

if __name__ == "__main__":
    main()
