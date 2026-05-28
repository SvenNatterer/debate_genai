import os
from PIL import Image
import numpy as np

SRC = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/media__1779875861495.png"

CELL_W = 130
IDLE_Y1 = 52
IDLE_Y2 = 170
NUM_COLS = 6

# Exact cell x-boundaries measured from dark separator columns
# Dark cols found at: 131-133, 263-264, 395-396, 526-528, 659, 791-792
CELL_BOUNDS = [
    (0,   130),   # col 0
    (134, 262),   # col 1
    (265, 394),   # col 2
    (397, 525),   # col 3
    (529, 658),   # col 4
    (660, 790),   # col 5
]

def extract_and_clean_cells(img):
    cells = []
    for (x1, x2) in CELL_BOUNDS:
        y1 = IDLE_Y1
        y2 = IDLE_Y2

        cell = img.crop((x1, y1, x2, y2)).convert("RGBA")
        arr = np.array(cell)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

        # Remove green background
        g_f = g.astype(float)
        green_mask = (g > 100) & (g_f - r > 40) & (g_f - b > 40)

        # Remove dark cell border lines at the cell edges (separator lines)
        dark_mask = (r.astype(int) + g.astype(int) + b.astype(int)) < 60

        border_mask = np.zeros_like(green_mask)
        border_mask[0:2, :] = True   # top border
        border_mask[-2:, :] = True   # bottom border
        border_mask[:, 0:4] = True   # left border
        border_mask[:, -4:] = True   # right border

        remove_mask = green_mask | (dark_mask & border_mask)
        arr[remove_mask, 3] = 0

        # Despill: suppress green fringe at edges
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

        cells.append(Image.fromarray(arr))
    return cells


def generate_portrait(frame, output_path):
    arr = np.array(frame)
    y_idx, x_idx = np.where(arr[:, :, 3] > 0)
    if len(y_idx) == 0:
        print("Error: empty frame for portrait!")
        return
    ymin, ymax = y_idx.min(), y_idx.max()
    xmin, xmax = x_idx.min(), x_idx.max()
    cropped = frame.crop((xmin, ymin, xmax + 1, ymax + 1))

    h_target = 180
    w_target = int(round(cropped.width * h_target / cropped.height))
    scaled = cropped.resize((w_target, h_target), Image.Resampling.NEAREST)

    canvas = Image.new("RGBA", (256, 256), (40, 64, 86, 255))
    x_off = (256 - w_target) // 2
    y_off = 256 - h_target - 20
    canvas.paste(scaled, (x_off, y_off), scaled)
    canvas.convert("RGB").save(output_path)
    print(f"Saved portrait → {output_path}")


def generate_spritesheet(frames, output_path, frame_indices):
    H_TARGET = 148
    Y_ANCHOR = 171
    sheet_frames = []

    for src_idx in frame_indices:
        frame = frames[src_idx]
        arr = np.array(frame)
        y_idx, x_idx = np.where(arr[:, :, 3] > 0)

        if len(y_idx) == 0:
            print(f"Warning: frame {src_idx} is empty!")
            sheet_frames.append(Image.new("RGBA", (176, 176), (0, 0, 0, 0)))
            continue

        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        cropped = frame.crop((xmin, ymin, xmax + 1, ymax + 1))

        w_scaled = int(round(cropped.width * H_TARGET / cropped.height))
        scaled = cropped.resize((w_scaled, H_TARGET), Image.Resampling.NEAREST)

        canvas = Image.new("RGBA", (176, 176), (0, 0, 0, 0))
        x_off = (176 - w_scaled) // 2
        y_off = Y_ANCHOR - H_TARGET
        canvas.paste(scaled, (x_off, y_off), scaled)
        sheet_frames.append(canvas)

    n = len(sheet_frames)
    spritesheet = Image.new("RGBA", (176 * n, 176), (0, 0, 0, 0))
    for i, f in enumerate(sheet_frames):
        spritesheet.paste(f, (i * 176, 0))
    spritesheet.save(output_path)
    print(f"Saved spritesheet ({n} frames) → {output_path}")


def main():
    os.makedirs("images", exist_ok=True)
    img = Image.open(SRC)
    frames = extract_and_clean_cells(img)

    # Use frames 0,1,2,3,4 (all 5 look good; skip frame 5 which is nearly empty)
    generate_portrait(frames[0], "images/Locke.png")
    generate_spritesheet(frames, "images/Locke_idle_strip.png", [0, 1, 2, 3, 4])


if __name__ == "__main__":
    main()
