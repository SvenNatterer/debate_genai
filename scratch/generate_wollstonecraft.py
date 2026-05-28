import os
from PIL import Image
import numpy as np

def extract_and_clean_cells(img, start_y):
    cells = []
    boundaries = [0, 133, 266, 399, 532, 665]
    for col in range(5):
        x1 = boundaries[col]
        x2 = boundaries[col + 1]
        y1 = start_y
        y2 = start_y + 120  # generous height of 120px to prevent clipping
        
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
    print(f"Saved spritesheet to {output_path}")

def generate_preview_gif(spritesheet_path, output_path):
    img = Image.open(spritesheet_path)
    frames = []
    bg_color = (15, 22, 38, 255)  # #0f1626
    
    for idx in range(5):
        frame = img.crop((idx * 176, 0, (idx + 1) * 176, 176))
        canvas = Image.new("RGBA", (176, 176), bg_color)
        canvas.paste(frame, (0, 0), frame)
        frames.append(canvas.convert("RGB"))
        
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    print(f"Saved animated preview GIF to {output_path}")

def main():
    spritesheet_path = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779874616421.png"
    if not os.path.exists(spritesheet_path):
        print(f"Error: Wollstonecraft source image not found at {spritesheet_path}")
        return
        
    img = Image.open(spritesheet_path)
    
    os.makedirs("images", exist_ok=True)
    
    print("\n--- Processing Mary Wollstonecraft ---")
    wollstonecraft_frames = extract_and_clean_cells(img, 53)
    generate_portrait(wollstonecraft_frames[0], "images/Wollstonecraft.png")
    generate_spritesheet(wollstonecraft_frames, "images/Wollstonecraft_idle_strip.png")
    generate_preview_gif(
        "images/Wollstonecraft_idle_strip.png",
        "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/Wollstonecraft.gif"
    )

if __name__ == "__main__":
    main()
