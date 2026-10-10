import argparse
import concurrent.futures
import os
from pathlib import Path

from PIL import Image
from rembg import remove


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def process_image(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as image:
        image = image.convert("RGBA")
        result = remove(image)
        result = result.convert("RGBA")

        white_bg = Image.new("RGB", result.size, (255, 255, 255))
        white_bg.paste(result, mask=result.split()[3])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        white_bg.save(output_path, optimize=True)


def get_image_files(folder: Path):
    for root, _, files in os.walk(folder):
        for file_name in files:
            file_path = Path(root) / file_name
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield file_path


def main():
    parser = argparse.ArgumentParser(description="Bulk remove image backgrounds and replace with white.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset"),
        help="Input folder containing class subfolders or images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cleaned_dataset"),
        help="Output folder for cleaned images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker threads to use.",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(get_image_files(input_dir))

    if not image_paths:
        print(f"No supported image files found in {input_dir}")
        return

    print(f"Processing {len(image_paths)} images from {input_dir}")
    print(f"Writing cleaned images to {output_dir}")
    print(f"Using {args.workers} worker threads")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = []
        for input_path in image_paths:
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path = output_path.with_suffix(".png")
            tasks.append(executor.submit(process_image, input_path, output_path))

        for task in concurrent.futures.as_completed(tasks):
            task.result()

    print(f"Finished processing. Cleaned images saved to {output_dir}")


if __name__ == "__main__":
    main()
