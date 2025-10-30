import sys, subprocess
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python downsize_video.py {video}.mp4")
    sys.exit(1)

inp = Path(sys.argv[1])
outp = inp.with_name(f"{inp.stem}_downsize.mp4")

cmd = [
    "ffmpeg", "-y", "-i", str(inp),
    "-vf", "scale=320:180:flags=lanczos",        # replace with the pad pipeline below to keep aspect
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k",
    "-movflags", "+faststart",
    str(outp)
]
# To preserve aspect ratio with padding instead, use:
# cmd[6] = "scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2:color=black"

subprocess.run(cmd, check=True)
print(f"Wrote: {outp}")
