"""
Extract idle frames from the new Descartes and Leibniz sprite sheets
and generate portrait + idle strip images.

These sprite sheets use transparent backgrounds (no checkerboard),
so we only need to crop individual cells and assemble them.
"""
import os
from collections import deque
from PIL import Image
import numpy as np


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


def find_cell_boundaries(arr, idle_y1, idle_y2):
    """Find cell x-boundaries in the idle row using transparency gaps."""
    idle_area = arr[idle_y1:idle_y2, :, :]
    idle_h = idle_area.shape[0]
    w = arr.shape[1]
    alpha = idle_area[:, :, 3]

    # Find columns that are mostly transparent (gaps between cells)
    col_transparent = (alpha == 0).sum(axis=0)
    gap_threshold = idle_h * 0.8
    gap_cols = np.where(col_transparent > gap_threshold)[0]

    if len(gap_cols) == 0:
        return []

    # Group consecutive gap columns
    gaps = []
    g_start = gap_cols[0]
    for i in range(1, len(gap_cols)):
        if gap_cols[i] - gap_cols[i - 1] > 2:
            gaps.append((int(g_start), int(gap_cols[i - 1])))
            g_start = gap_cols[i]
    gaps.append((int(g_start), int(gap_cols[-1])))

    # Content regions between gaps (cells)
    cells = []
    for i in range(len(gaps) - 1):
        cell_start = gaps[i][1] + 1
        cell_end = gaps[i + 1][0] - 1
        if cell_end - cell_start > 20:
            cells.append((cell_start, cell_end))

    return cells


def find_grid_line_boundaries(arr, idle_y1, idle_y2):
    """Find cell x-boundaries using grid lines (low-std columns)."""
    idle_area = arr[idle_y1:idle_y2, :, :]
    idle_h = idle_area.shape[0]
    w = arr.shape[1]
    
    rgb = idle_area[:, :, :3].astype(float)
    col_std = rgb.std(axis=0).mean(axis=1)
    
    # Find thin low-std columns that are grid lines
    low_std_cols = np.where(col_std < 5)[0]
    
    if len(low_std_cols) == 0:
        return []
    
    # Group consecutive columns
    groups = []
    g_start = low_std_cols[0]
    for i in range(1, len(low_std_cols)):
        if low_std_cols[i] - low_std_cols[i - 1] > 2:
            groups.append((int(g_start), int(low_std_cols[i - 1])))
            g_start = low_std_cols[i]
    groups.append((int(g_start), int(low_std_cols[-1])))
    
    # Filter to single-pixel or thin lines that are actual grid separators
    # (not the edges of the image)
    thin_lines = [g for g in groups if g[1] - g[0] <= 2 and g[0] > 10 and g[0] < w - 20]
    
    # Content regions between grid lines
    cells = []
    all_boundaries = [(0, groups[0][1])] + thin_lines + [(groups[-1][0], w - 1)]
    for i in range(len(all_boundaries) - 1):
        cell_start = all_boundaries[i][1] + 1
        cell_end = all_boundaries[i + 1][0] - 1
        if cell_end - cell_start > 30:
            cells.append((cell_start, cell_end))
    
    return cells


def extract_idle_cells(img_path, idle_y1, idle_y2, method="transparency"):
    """Extract idle frame cells from the sprite sheet."""
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    
    if method == "transparency":
        cells_bounds = find_cell_boundaries(arr, idle_y1, idle_y2)
    else:
        cells_bounds = find_grid_line_boundaries(arr, idle_y1, idle_y2)
    
    cells = []
    for x1, x2 in cells_bounds:
        cell_arr = arr[idle_y1:idle_y2, x1:x2 + 1, :].copy()
        # Clean small artifacts
        cell_arr = remove_small_alpha_components(cell_arr)
        cell_arr = keep_only_largest_component(cell_arr)
        cells.append(Image.fromarray(cell_arr))
    
    return cells


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


def main():
    os.makedirs("images", exist_ok=True)

    philosophers = [
        {
            "name": "Descartes",
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779895651406.png",
            "idle_y1": 55,
            "idle_y2": 180,
            "h_target": 148,
            "method": "transparency",
        },
        {
            "name": "Leibniz",
            "img_path": "/Users/svennatterer/.gemini/antigravity/brain/a725d1b2-2fc0-4c12-8323-c5a1ddcd4d28/media__1779895654167.png",
            "idle_y1": 64,
            "idle_y2": 177,
            "h_target": 148,
            "method": "transparency",
        },
    ]

    for p in philosophers:
        print(f"\n--- Processing {p['name']} ---")
        if not os.path.exists(p["img_path"]):
            print(f"Error: Source image not found at {p['img_path']}")
            continue

        cells = extract_idle_cells(
            p["img_path"], p["idle_y1"], p["idle_y2"], method=p["method"]
        )
        print(f"  Found {len(cells)} idle cells")

        if len(cells) < 5:
            print(f"  Error: Need at least 5 cells, found {len(cells)}")
            continue

        # Use only the first 5 frames for the idle strip
        frames = cells[:5]

        generate_portrait(frames[0], f"images/{p['name']}.png")
        generate_spritesheet(frames, p["h_target"], f"images/{p['name']}_idle_strip.png")


if __name__ == "__main__":
    main()
