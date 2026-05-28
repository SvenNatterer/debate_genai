import os
from PIL import Image
import numpy as np

def remove_background(img_path, threshold=30.0):
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    h, w, _ = arr.shape
    
    # Background color at top-left corner
    bg_color = arr[0, 0, :3].astype(float)
    
    # BFS starting from all border pixels to find connected background pixels
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    
    # Add borders
    for y in range(h):
        queue.append((y, 0))
        queue.append((y, w - 1))
        visited[y, 0] = True
        visited[y, w - 1] = True
    for x in range(1, w - 1):
        queue.append((0, x))
        queue.append((h - 1, x))
        visited[0, x] = True
        visited[h - 1, x] = True
        
    head = 0
    while head < len(queue):
        cy, cx = queue[head]
        head += 1
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                diff = np.linalg.norm(arr[ny, nx, :3].astype(float) - bg_color)
                if diff < threshold:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
                    
    # Make background pixels transparent
    arr[visited, 3] = 0
    return Image.fromarray(arr)

def generate_spritesheet(portrait_path, output_path):
    print(f"Processing {portrait_path}...")
    # 1. Remove background
    cleared = remove_background(portrait_path)
    
    # 2. Find bounding box of non-transparent character
    arr = np.array(cleared)
    y_indices, x_indices = np.where(arr[:, :, 3] > 0)
    
    if len(y_indices) == 0:
        print(f"Error: No character found in {portrait_path}!")
        return
        
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Crop character
    cropped = cleared.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    # 3. Base scaling (target height = 151)
    h_orig = cropped.height
    w_orig = cropped.width
    
    h_base = 151
    w_base = int(round(w_orig * (h_base / h_orig)))
    
    # Base resize using Lanczos for high quality details
    base_resized = cropped.resize((w_base, h_base), Image.Resampling.LANCZOS)
    
    # Create the 5 frames of the spritesheet
    frames = []
    # Animation factors matching Nietzsche's strip
    # Frame 0: 100% height, 100% width
    # Frame 1: 95.3% height, 98% width
    # Frame 2: 93.3% height, 96% width
    # Frame 3: 100% height, 100% width
    # Frame 4: 100% height, 100% width
    scaling_factors = [
        (1.0, 1.0),
        (0.98, 0.953),
        (0.96, 0.933),
        (1.0, 1.0),
        (1.0, 1.0)
    ]
    
    y_anchor = 171  # Floor anchor line
    
    for i, (w_factor, h_factor) in enumerate(scaling_factors):
        frame_w = int(round(w_base * w_factor))
        frame_h = int(round(h_base * h_factor))
        
        # Resize to frame dimensions
        frame_char = base_resized.resize((frame_w, frame_h), Image.Resampling.LANCZOS)
        
        # Binary thresholding on alpha channel to keep clean sharp pixel-art edges
        char_arr = np.array(frame_char)
        char_arr[char_arr[:, :, 3] <= 128, 3] = 0
        char_arr[char_arr[:, :, 3] > 128, 3] = 255
        frame_char_thresholded = Image.fromarray(char_arr)
        
        # Create empty 176x176 frame
        frame_canvas = Image.new("RGBA", (176, 176), (0, 0, 0, 0))
        
        # Align: center horizontally, anchor bottom at y_anchor
        x_offset = (176 - frame_w) // 2
        y_offset = y_anchor - frame_h
        
        frame_canvas.paste(frame_char_thresholded, (x_offset, y_offset), frame_char_thresholded)
        frames.append(frame_canvas)
        
    # Stitch frames side-by-side
    spritesheet = Image.new("RGBA", (880, 176), (0, 0, 0, 0))
    for idx, frame in enumerate(frames):
        spritesheet.paste(frame, (idx * 176, 0))
        
    # Save the output file
    spritesheet.save(output_path)
    print(f"Saved spritesheet to {output_path}")

def main():
    portraits = [
        ("images/Socrates.png", "images/Socrates_idle_strip.png"),
        ("images/Plato.png", "images/Plato_idle_strip.png"),
        ("images/Aristotle.png", "images/Aristotle_idle_strip.png"),
        ("images/Kant.png", "images/Kant_idle_strip.png"),
        ("images/Mill.png", "images/Mill_idle_strip.png"),
        ("images/Beauvoir.png", "images/Beauvoir_idle_strip.png"),
    ]
    
    os.makedirs("images", exist_ok=True)
    
    for portrait, output in portraits:
        if os.path.exists(portrait):
            generate_spritesheet(portrait, output)
        else:
            print(f"Skipping {portrait} - file not found.")

if __name__ == "__main__":
    main()
