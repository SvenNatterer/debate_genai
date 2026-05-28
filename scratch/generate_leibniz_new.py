import os
from collections import deque
from PIL import Image
import numpy as np

CELL_BOUNDS = [
    (0, 130),
    (132, 261),
    (262, 392),
    (393, 522),
    (524, 653),
]
IDLE_Y1 = 49
IDLE_Y2 = 167


def keep_largest_alpha_component(arr):
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    seen = np.zeros((h, w), dtype=bool)
    best_component = []

    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0 or seen[y, x]:
                continue

            component = []
            queue = deque([(y, x)])
            seen[y, x] = True

            while queue:
                current_y, current_x = queue.popleft()
                component.append((current_y, current_x))

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        next_y = current_y + dy
                        next_x = current_x + dx
                        if (
                            0 <= next_y < h
                            and 0 <= next_x < w
                            and alpha[next_y, next_x] > 0
                            and not seen[next_y, next_x]
                        ):
                            seen[next_y, next_x] = True
                            queue.append((next_y, next_x))

            if len(component) > len(best_component):
                best_component = component

    if best_component:
        keep = np.zeros((h, w), dtype=bool)
        ys, xs = zip(*best_component)
        keep[list(ys), list(xs)] = True
        arr[~keep, 3] = 0

    return arr


def remove_connected_green_screen(cell):
    arr = np.array(cell.convert("RGBA"))
    rgb = arr[:, :, :3].astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, w = r.shape

    # Only remove true green-screen pixels connected to the cell border. This
    # preserves olive/brown clothing, dark outlines, hair, and facial pixels.
    bg_candidate = (g > 105) & (g - r > 35) & (g - b > 35)

    # Crop separator residue may be black; remove it only when it touches edges.
    dark = (r + g + b) < 75
    edge = np.zeros((h, w), dtype=bool)
    edge[:2, :] = True
    edge[-2:, :] = True
    edge[:, :2] = True
    edge[:, -2:] = True
    bg_candidate |= dark & edge

    background = np.zeros((h, w), dtype=bool)
    queue = deque()
    for x in range(w):
        for y in (0, h - 1):
            if bg_candidate[y, x] and not background[y, x]:
                background[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if bg_candidate[y, x] and not background[y, x]:
                background[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < h
                and 0 <= next_x < w
                and bg_candidate[next_y, next_x]
                and not background[next_y, next_x]
            ):
                background[next_y, next_x] = True
                queue.append((next_y, next_x))

    arr[background, 3] = 0
    arr = keep_largest_alpha_component(arr)

    # Despill remaining edge pixels instead of deleting them.
    alpha = arr[:, :, 3]
    alpha_pad = np.pad(alpha, 1, mode="constant", constant_values=0)
    edge_visible = (alpha > 0) & (
        (alpha_pad[2:, 1:-1] == 0)
        | (alpha_pad[:-2, 1:-1] == 0)
        | (alpha_pad[1:-1, 2:] == 0)
        | (alpha_pad[1:-1, :-2] == 0)
    )
    edge_pad = np.pad(edge_visible, 1, mode="constant", constant_values=False)
    edge_dilated = (
        edge_visible
        | edge_pad[2:, 1:-1]
        | edge_pad[:-2, 1:-1]
        | edge_pad[1:-1, 2:]
        | edge_pad[1:-1, :-2]
    )
    rr, gg, bb = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_edge = edge_dilated & (gg > rr) & (gg > bb)
    arr[green_edge, 1] = np.maximum(rr[green_edge], bb[green_edge])

    return Image.fromarray(arr)


def extract_and_clean_idle_frames(img_path):
    img = Image.open(img_path)
    frames = []
    for x1, x2 in CELL_BOUNDS:
        cell = img.crop((x1, IDLE_Y1, x2, IDLE_Y2))
        frames.append(remove_connected_green_screen(cell))
        
    return frames

def generate_portrait(frame0, output_path):
    # Use alpha channel to find the character bounding box (avoids masking dark pixels)
    arr = np.array(frame0)
    y_idx, x_idx = np.where(arr[:, :, 3] > 0)
    
    if len(y_idx) == 0:
        print("Error: Empty frame passed for portrait generation!")
        return
        
    ymin, ymax = y_idx.min(), y_idx.max()
    xmin, xmax = x_idx.min(), x_idx.max()
    
    cropped_char = frame0.crop((xmin, ymin, xmax + 1, ymax + 1))
    
    # Scale character to height=180 using NEAREST to keep pixel art crisp
    h_target = 180
    w_target = int(round(cropped_char.width * (h_target / cropped_char.height)))
    scaled_char = cropped_char.resize((w_target, h_target), Image.Resampling.NEAREST)
    
    # Create square 256x256 background with theme color (40, 64, 86)
    portrait_canvas = Image.new("RGBA", (256, 256), (40, 64, 86, 255))
    
    # Paste centered, anchored near bottom (20px padding at bottom)
    x_offset = (256 - w_target) // 2
    y_offset = 256 - h_target - 20
    
    portrait_canvas.paste(scaled_char, (x_offset, y_offset), scaled_char)
    portrait_canvas.convert("RGB").save(output_path)
    print(f"Saved static portrait to {output_path}")

def generate_spritesheet(frames, output_path):
    h_target = 148
    y_anchor = 171
    
    sheet_frames = []
    for idx, frame in enumerate(frames):
        # Use alpha channel for bounding box (avoids masking dark character pixels)
        arr = np.array(frame)
        y_idx, x_idx = np.where(arr[:, :, 3] > 0)
        
        if len(y_idx) == 0:
            print(f"Warning: Frame {idx} is empty!")
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
    img_path = "/Users/svennatterer/.gemini/antigravity/brain/8652b6c1-ec58-45b0-a140-06752430a565/media__1779873959867.png"
    if not os.path.exists(img_path):
        print(f"Error: Source image not found at {img_path}")
        return
        
    print("\n--- Extracting Leibniz from new spritesheet ---")
    frames = extract_and_clean_idle_frames(img_path)
    
    # Generate files
    generate_portrait(frames[0], "images/Leibniz.png")
    generate_spritesheet(frames, "images/Leibniz_idle_strip.png")

if __name__ == "__main__":
    main()
