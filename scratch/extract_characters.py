import os
import numpy as np
from PIL import Image

def remove_background(img, threshold=8):
    # Convert image to RGBA
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    h, w, _ = arr.shape
    
    # BFS starting from all border pixels to find connected background checkerboard pixels
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    
    # Add borders to queue
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
                r, g, b, a = arr[ny, nx]
                # Check if it matches the checkerboard pattern (light grey or white)
                # It should be grayscaled (R, G, B very close) and relatively light (>170)
                is_bg = (170 <= r <= 235) and (170 <= g <= 235) and (170 <= b <= 235) and (abs(int(r) - g) <= threshold) and (abs(int(g) - b) <= threshold)
                if is_bg:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
                    
    # Set alpha of background to 0
    arr[visited, 3] = 0
    return Image.fromarray(arr)

def process_character(img_path, portrait_box, y_start, y_end, name_lower, out_name):
    print(f"\nProcessing {out_name}...")
    img = Image.open(img_path)
    
    # 1. Process portrait
    portrait_crop = img.crop(portrait_box)
    # Remove background from portrait as well, if checkerboard is present
    portrait_transparent = remove_background(portrait_crop, threshold=12)
    # Resize portrait to 256x256 using nearest neighbor to preserve pixel art
    portrait_final = portrait_transparent.resize((256, 256), Image.Resampling.NEAREST)
    portrait_path = f"images/{out_name}.png"
    portrait_final.save(portrait_path)
    print(f"Saved portrait to {portrait_path}")
    
    # 2. Extract and process 5 idle frames
    frames = []
    # Spacing parameters matching the 142px width grid starting at X=15
    col_width = 142
    start_x = 15
    
    for c in range(5):
        x1 = start_x + c * col_width
        x2 = start_x + (c + 1) * col_width
        frame_crop = img.crop((x1, y_start, x2, y_end))
        # Remove checkerboard background
        frame_transparent = remove_background(frame_crop, threshold=8)
        frames.append(frame_transparent)
        
    # Find bounding box of non-transparent character in each frame
    bboxes = []
    for idx, f in enumerate(frames):
        arr = np.array(f)
        y_indices, x_indices = np.where(arr[:, :, 3] > 0)
        if len(y_indices) > 0:
            bboxes.append((x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()))
        else:
            # Fallback if frame is empty (should not happen)
            bboxes.append((0, 0, f.width - 1, f.height - 1))
            
    # Compute union bounding box to keep relative motion intact
    u_xmin = min(bbox[0] for bbox in bboxes)
    u_ymin = min(bbox[1] for bbox in bboxes)
    u_xmax = max(bbox[2] for bbox in bboxes)
    u_ymax = max(bbox[3] for bbox in bboxes)
    
    print(f"Union bounding box: X=[{u_xmin}, {u_xmax}], Y=[{u_ymin}, {u_ymax}]")
    
    # Target height is 151
    h_base = 151
    y_anchor = 171
    
    processed_frames = []
    for f in frames:
        # Crop to the union bounding box
        cropped_f = f.crop((u_xmin, u_ymin, u_xmax + 1, u_ymax + 1))
        # Determine target width to maintain aspect ratio
        w_target = int(round(cropped_f.width * (h_base / cropped_f.height)))
        # Resize using nearest neighbor
        resized_f = cropped_f.resize((w_target, h_base), Image.Resampling.NEAREST)
        
        # Place inside empty 176x176 canvas
        canvas = Image.new("RGBA", (176, 176), (0, 0, 0, 0))
        x_offset = (176 - w_target) // 2
        y_offset = y_anchor - h_base
        
        canvas.paste(resized_f, (x_offset, y_offset), resized_f)
        processed_frames.append(canvas)
        
    # Stitch 5 frames side-by-side
    spritesheet = Image.new("RGBA", (880, 176), (0, 0, 0, 0))
    for idx, f in enumerate(processed_frames):
        spritesheet.paste(f, (idx * 176, 0))
        
    spritesheet_path = f"images/{out_name}_idle_strip.png"
    spritesheet.save(spritesheet_path)
    print(f"Saved idle strip to {spritesheet_path}")

def main():
    os.makedirs("images", exist_ok=True)
    
    # Philosopher data: image path, portrait box, y_start, y_end, name key, filename key
    philosophers = [
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779891364685.jpg",
            "portrait_box": (26, 6, 89, 69),
            "y_start": 70,
            "y_end": 189,
            "name": "de_beauvoir",
            "out_name": "Beauvoir"
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779891367088.jpg",
            "portrait_box": (17, 1, 71, 55),
            "y_start": 56,
            "y_end": 182,
            "name": "sartre",
            "out_name": "Sartre"
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779891375258.jpg",
            "portrait_box": (0, 0, 60, 60),
            "y_start": 61,
            "y_end": 187,
            "name": "arendt",
            "out_name": "Arendt"
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779892440601.jpg",
            "portrait_box": (14, 6, 72, 64),
            "y_start": 70,
            "y_end": 188,
            "name": "mill",
            "out_name": "Mill"
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779892884361.jpg",
            "portrait_box": (26, 4, 94, 72),
            "y_start": 70,
            "y_end": 188,
            "name": "marx",
            "out_name": "Marx"
        }
    ]
    
    for p in philosophers:
        process_character(
            p["img_path"], 
            p["portrait_box"], 
            p["y_start"], 
            p["y_end"], 
            p["name"], 
            p["out_name"]
        )

if __name__ == "__main__":
    main()
