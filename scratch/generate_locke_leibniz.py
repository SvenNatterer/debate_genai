import os
from PIL import Image
import numpy as np

def extract_and_clean_cells(img, start_x, start_y):
    cells = []
    for col in range(4):
        x1 = start_x + col * 64
        x2 = start_x + (col + 1) * 64
        y1 = start_y
        y2 = start_y + 48
        
        cell = img.crop((x1, y1, x2, y2)).convert("RGBA")
        arr = np.array(cell)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        
        # Chroma key for green background
        g_f = g.astype(float)
        green_mask = (g > 120) & (g_f - r > 50) & (g_f - b > 50)
        
        # Remove dark cell borders near the cell edges
        dark_mask = (r < 75) & (g < 75) & (b < 75)
        border_mask = np.zeros_like(green_mask)
        border_mask[0:2, :] = True
        border_mask[-2:, :] = True
        border_mask[:, 0:2] = True
        border_mask[:, -2:] = True
        
        remove_mask = green_mask | (dark_mask & border_mask)
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

def generate_portrait(frame0, output_path):
    # 1. Bounding box of character in frame0
    arr = np.array(frame0)
    y_idx, x_idx = np.where(arr[:, :, 3] > 0)
    
    if len(y_idx) == 0:
        print("Error: Empty frame passed for portrait generation!")
        return
        
    ymin, ymax = y_idx.min(), y_idx.max()
    xmin, xmax = x_idx.min(), x_idx.max()
    
    cropped_char = frame0.crop((xmin, ymin, xmax + 1, ymax + 1))
    
    # 2. Scale character to height=180 using NEAREST to keep pixel art crisp
    h_target = 180
    w_target = int(round(cropped_char.width * (h_target / cropped_char.height)))
    scaled_char = cropped_char.resize((w_target, h_target), Image.Resampling.NEAREST)
    
    # 3. Create square 256x256 background with theme color (40, 64, 86)
    portrait_canvas = Image.new("RGBA", (256, 256), (40, 64, 86, 255))
    
    # 4. Paste centered, anchored near bottom (20px padding at bottom)
    x_offset = (256 - w_target) // 2
    y_offset = 256 - h_target - 20
    
    portrait_canvas.paste(scaled_char, (x_offset, y_offset), scaled_char)
    portrait_canvas.convert("RGB").save(output_path)
    print(f"Saved static portrait to {output_path}")

def generate_spritesheet(frames, output_path):
    h_target = 148
    y_anchor = 171
    
    sheet_frames = []
    # Ping-pong sequence: Frame 0, 1, 2, 3, 2
    sequence = [0, 1, 2, 3, 2]
    
    for idx in sequence:
        frame = frames[idx]
        
        # Bounding box of the character in this frame
        arr = np.array(frame)
        y_idx, x_idx = np.where(arr[:, :, 3] > 0)
        
        if len(y_idx) == 0:
            print(f"Warning: Frame {idx} is empty!")
            # Fallback to empty 176x176 frame
            sheet_frames.append(Image.new("RGBA", (176, 176), (0, 0, 0, 0)))
            continue
            
        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        
        cropped_char = frame.crop((xmin, ymin, xmax + 1, ymax + 1))
        
        # Scale proportionally to height=148
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
    print(f"Saved spritesheet strip to {output_path}")

def main():
    spritesheet_path = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779869597424.jpg"
    if not os.path.exists(spritesheet_path):
        print(f"Error: Spritesheet compilation not found at {spritesheet_path}")
        return
        
    img = Image.open(spritesheet_path)
    
    os.makedirs("images", exist_ok=True)
    
    # 1. John Locke: starts at X=0, Y=91 in the original image (Row 2, Left)
    print("\n--- Processing John Locke ---")
    locke_frames = extract_and_clean_cells(img, 0, 91)
    generate_portrait(locke_frames[0], "images/Locke.png")
    generate_spritesheet(locke_frames, "images/Locke_idle_strip.png")
    
    # 2. Leibniz: starts at X=768, Y=91 in the original image (Row 2, Right)
    print("\n--- Processing Gottfried Wilhelm Leibniz ---")
    leibniz_frames = extract_and_clean_cells(img, 768, 91)
    generate_portrait(leibniz_frames[0], "images/Leibniz.png")
    generate_spritesheet(leibniz_frames, "images/Leibniz_idle_strip.png")

if __name__ == "__main__":
    main()
