import os
from collections import deque
from PIL import Image
import numpy as np

BACKGROUND_SATURATION_LIMIT = 15
MIN_COMPONENT_PIXELS = 24

def remove_small_alpha_components(arr):
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    seen = np.zeros((h, w), dtype=bool)

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

            if len(component) < MIN_COMPONENT_PIXELS:
                ys, xs = zip(*component)
                arr[list(ys), list(xs), 3] = 0

    return arr

def keep_only_largest_component(arr):
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if alpha[y, x] == 0 or seen[y, x]:
                continue
            comp = []
            q = deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and alpha[ny, nx] > 0 and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            components.append(comp)

    if not components:
        return arr

    largest_comp = max(components, key=len)
    new_alpha = np.zeros((h, w), dtype=np.uint8)
    for y, x in largest_comp:
        new_alpha[y, x] = arr[y, x, 3]

    arr[:, :, 3] = new_alpha
    return arr

def remove_connected_checkerboard_background(cell, sat_limit=15):
    arr = np.array(cell.convert("RGBA"))
    h, w, _ = arr.shape

    # Apply border mask to clear grid lines and artifacts near cell borders (shaving 8px horizontal, 6px vertical)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:, 0:8] = True
    border_mask[:, -8:] = True
    border_mask[0:6, :] = True
    border_mask[-6:, :] = True
    arr[border_mask, 3] = 0

    rgb = arr[:, :, :3].astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_channel = np.maximum.reduce([r, g, b])
    min_channel = np.minimum.reduce([r, g, b])
    avg = (r + g + b) / 3.0

    bg_candidate = (
        (max_channel - min_channel <= sat_limit)
        & (avg > 85)
    )
    bg_candidate |= border_mask

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
    arr = remove_small_alpha_components(arr)
    arr = keep_only_largest_component(arr)

    return Image.fromarray(arr)

def generate_portrait(frame0, output_path):
    arr = np.array(frame0)
    y_idx, x_idx = np.where(arr[:, :, 3] > 0)
    
    if len(y_idx) == 0:
        print("Error: Empty frame passed for portrait generation!")
        return
        
    ymin, ymax = y_idx.min(), y_idx.max()
    xmin, xmax = x_idx.min(), x_idx.max()
    
    cropped_char = frame0.crop((xmin, ymin, xmax + 1, ymax + 1))
    
    h_target = 180
    w_target = int(round(cropped_char.width * (h_target / cropped_char.height)))
    scaled_char = cropped_char.resize((w_target, h_target), Image.Resampling.NEAREST)
    
    portrait_canvas = Image.new("RGBA", (256, 256), (40, 64, 86, 255))
    
    x_offset = (256 - w_target) // 2
    y_offset = 256 - h_target - 20
    
    portrait_canvas.paste(scaled_char, (x_offset, y_offset), scaled_char)
    portrait_canvas.convert("RGB").save(output_path)
    print(f"Saved static portrait to {output_path}")

def generate_spritesheet(frames, h_target, output_path):
    y_anchor = 171
    
    sheet_frames = []
    
    for idx in range(5):
        frame = frames[idx]
        arr = np.array(frame)
        y_idx, x_idx = np.where(arr[:, :, 3] > 0)
        
        if len(y_idx) == 0:
            print(f"Warning: Frame {idx} is empty!")
            sheet_frames.append(Image.new("RGBA", (176, 176), (0, 0, 0, 0)))
            continue
            
        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        
        cropped_char = frame.crop((xmin, ymin, xmax + 1, ymax + 1))
        
        w_scaled = int(round(cropped_char.width * (h_target / cropped_char.height)))
        scaled_char = cropped_char.resize((w_scaled, h_target), Image.Resampling.NEAREST)
        
        canvas = Image.new("RGBA", (176, 176), (0, 0, 0, 0))
        
        x_offset = (176 - w_scaled) // 2
        y_offset = y_anchor - h_target
        
        canvas.paste(scaled_char, (x_offset, y_offset), scaled_char)
        sheet_frames.append(canvas)
        
    spritesheet = Image.new("RGBA", (880, 176), (0, 0, 0, 0))
    for idx, f in enumerate(sheet_frames):
        spritesheet.paste(f, (idx * 176, 0))
        
    spritesheet.save(output_path)
    print(f"Saved spritesheet to {output_path}")

def process_philosopher(p):
    print(f"\n--- Processing {p['out_name']} ---")
    if not os.path.exists(p['img_path']):
        print(f"Error: Source image not found at {p['img_path']}")
        return
        
    img = Image.open(p['img_path'])
    
    # Extract and clean 5 idle frames
    frames = []
    for x1, x2 in p['cell_bounds']:
        cell = img.crop((x1, p['y_start'], x2, p['y_end']))
        cleaned_cell = remove_connected_checkerboard_background(cell, sat_limit=p.get('sat_limit', 15))
        frames.append(cleaned_cell)
        
    generate_portrait(frames[0], f"images/{p['out_name']}.png")
    generate_spritesheet(frames, p['h_target'], f"images/{p['out_name']}_idle_strip.png")

def main():
    os.makedirs("images", exist_ok=True)
    
    # Define regular cell bounds for 142 width grid starting at X=15
    reg_bounds = [
        (15, 157),
        (157, 299),
        (299, 441),
        (441, 583),
        (583, 725)
    ]
    
    philosophers = [
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779874741386.png",
            "cell_bounds": [
                (0, 134),
                (136, 268),
                (270, 402),
                (404, 535),
                (538, 670)
            ],
            "y_start": 53,
            "y_end": 174,
            "h_target": 148,
            "out_name": "Kant",
            "sat_limit": 6
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779874741384.jpg",
            "cell_bounds": [
                (0, 134),
                (136, 269),
                (271, 404),
                (405, 537),
                (540, 673)
            ],
            "y_start": 54,
            "y_end": 175,
            "h_target": 148,
            "out_name": "Hegel",
            "sat_limit": 10
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779892440601.jpg",
            "cell_bounds": reg_bounds,
            "y_start": 70,
            "y_end": 188,
            "h_target": 151,
            "out_name": "Mill",
            "sat_limit": 15
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779892884361.jpg",
            "cell_bounds": reg_bounds,
            "y_start": 70,
            "y_end": 188,
            "h_target": 151,
            "out_name": "Marx",
            "sat_limit": 15
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/3b0044cb-d265-434a-a4c3-e1de411c39b9/media__1779889102279.jpg",
            "cell_bounds": [
                (0, 138),
                (140, 277),
                (279, 416),
                (418, 554),
                (556, 692)
            ],
            "y_start": 53,
            "y_end": 174,
            "h_target": 148,
            "out_name": "Wittgenstein",
            "sat_limit": 15
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779891375258.jpg",
            "cell_bounds": reg_bounds,
            "y_start": 61,
            "y_end": 187,
            "h_target": 151,
            "out_name": "Arendt",
            "sat_limit": 15
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779891367088.jpg",
            "cell_bounds": reg_bounds,
            "y_start": 56,
            "y_end": 182,
            "h_target": 151,
            "out_name": "Sartre",
            "sat_limit": 15
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779891364685.jpg",
            "cell_bounds": reg_bounds,
            "y_start": 70,
            "y_end": 189,
            "h_target": 151,
            "out_name": "Beauvoir",
            "sat_limit": 15
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/3b0044cb-d265-434a-a4c3-e1de411c39b9/media__1779889098701.jpg",
            "cell_bounds": [
                (77, 153),
                (155, 288),
                (290, 432),
                (434, 565),
                (567, 705)
            ],
            "y_start": 53,
            "y_end": 174,
            "h_target": 148,
            "out_name": "Foot",
            "sat_limit": 15
        },
        {
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/3b0044cb-d265-434a-a4c3-e1de411c39b9/media__1779889095294.jpg",
            "cell_bounds": [
                (71, 161),
                (163, 294),
                (296, 429),
                (431, 576),
                (578, 699)
            ],
            "y_start": 53,
            "y_end": 174,
            "h_target": 148,
            "out_name": "Butler",
            "sat_limit": 15
        }
    ]
    
    for p in philosophers:
        process_philosopher(p)

if __name__ == "__main__":
    main()
