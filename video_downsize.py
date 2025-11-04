import sys, subprocess
from pathlib import Path
import os

def downsize_video(input_path, output_path):
    """Downsize a single video file."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "scale=320:180:flags=lanczos",        # replace with the pad pipeline below to keep aspect
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]
    # To preserve aspect ratio with padding instead, use:
    # cmd[6] = "scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2:color=black"
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Processed: {input_path.name} -> {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {input_path.name}: {e}")
        return False

def process_directory(input_dir, output_dir):
    """Process all video files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    # Find all video files in the directory
    video_files = []
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print(f"No video files found in '{input_dir}'.")
        return
    
    print(f"Found {len(video_files)} video file(s) to process:")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    processed = 0
    failed = 0
    
    for video_file in video_files:
        output_file = output_path / f"{video_file.stem}_downsize{video_file.suffix}"
        
        # Skip if output file already exists
        if output_file.exists():
            print(f"⚠ Skipping {video_file.name}: output file already exists")
            continue
        
        if downsize_video(video_file, output_file):
            processed += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  ✓ Successfully processed: {processed} files")
    print(f"  ✗ Failed: {failed} files")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python video_downsize.py <input_directory> <output_directory>")
        print("Example: python video_downsize.py /path/to/input/videos /path/to/output/videos")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    process_directory(input_directory, output_directory)