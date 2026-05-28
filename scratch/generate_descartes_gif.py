import os
from PIL import Image

def main():
    spritesheet_path = "images/Descartes_idle_strip.png"
    if not os.path.exists(spritesheet_path):
        print("Error: Descartes spritesheet not found in images/")
        return
        
    img = Image.open(spritesheet_path)
    frames = []
    bg_color = (15, 22, 38, 255)  # #0f1626 matching the other preview GIFs
    
    for idx in range(5):
        frame = img.crop((idx * 176, 0, (idx + 1) * 176, 176))
        
        # Paste onto dark background
        canvas = Image.new("RGBA", (176, 176), bg_color)
        canvas.paste(frame, (0, 0), frame)
        frames.append(canvas.convert("RGB"))
        
    output_path = "/Users/svennatterer/.gemini/antigravity/brain/5972121d-a60c-4a9b-8aa1-59a16a901012/Descartes.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0
    )
    print(f"Saved animated preview GIF to {output_path}")

if __name__ == "__main__":
    main()
