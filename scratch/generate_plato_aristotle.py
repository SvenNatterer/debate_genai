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
        a_chan = arr[:, :, 3]
        a_pad = np.pad(a_chan, 1, mode="constant", constant_values=0)
        has_transparent_neighbor = (
            (a_pad[2:, 1:-1] == 0) |
            (a_pad[:-2, 1:-1] == 0) |
            (a_pad[1:-1, 2:] == 0) |
            (a_pad[1:-1, :-2] == 0)
        )
        edge_mask = (a_chan > 0) & has_transparent_neighbor
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

def generate_spritesheet(frames, output_path, frame_indices=None):
    h_target = 148
    y_anchor = 171
    
    if frame_indices is None:
        frame_indices = list(range(min(5, len(frames))))
    
    sheet_frames = []
    
    for src_idx in frame_indices:
        frame = frames[src_idx]
        
        # Bounding box of character in this frame
        arr = np.array(frame)
        y_idx, x_idx = np.where(arr[:, :, 3] > 0)
        
        if len(y_idx) == 0:
            print(f"Warning: Frame {src_idx} is empty!")
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
        
    # Stitch frames side-by-side
    n = len(sheet_frames)
    spritesheet = Image.new("RGBA", (176 * n, 176), (0, 0, 0, 0))
    for idx, f in enumerate(sheet_frames):
        spritesheet.paste(f, (idx * 176, 0))
        
    spritesheet.save(output_path)
    print(f"Saved spritesheet ({n} frames) to {output_path}")

def main():
    brain_dir = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012"
    
    jobs = [
        # Plato: only frames 1,2,3 have full hair; repeat them to fill 5 animation slots
        ("Plato",    "media__1779870752198.png", "images/Plato_idle_strip.png",    [1, 2, 3, 1, 2]),
        # Aristotle: use first 5 frames as before
        ("Aristotle","media__1779870755935.png", "images/Aristotle_idle_strip.png",[0, 1, 2, 3, 4]),
    ]
    
    os.makedirs("images", exist_ok=True)
    
    for name, src_file, dest_file, frame_indices in jobs:
        src_path = os.path.join(brain_dir, src_file)
        if not os.path.exists(src_path):
            print(f"Error: {name} source image not found at {src_path}")
            continue
            
        print(f"\n--- Processing {name} ---")
        img = Image.open(src_path)
        frames = extract_and_clean_cells(img, 53, num_cols=6)
        generate_spritesheet(frames, dest_file, frame_indices=frame_indices)

if __name__ == "__main__":
    main()
