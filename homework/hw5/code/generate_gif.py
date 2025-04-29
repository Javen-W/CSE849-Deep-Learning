import os
from PIL import Image


def create_gif_from_plots(input_dir, output_gif_path, duration=500):
    """
    Create a GIF from a directory of PNG images.
    Args:
        input_dir (str): Directory containing the PNG images.
        output_gif_path (str): Path to save the output GIF.
        duration (int): Duration of each frame in milliseconds (default: 500ms).
    """
    # Get list of PNG files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # Sort files to ensure correct order (assuming filenames allow numerical sorting)
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else x)

    # Check if there are any images
    if not image_files:
        raise ValueError(f"No PNG files found in directory: {input_dir}")

    # Open all images
    images = []
    for filename in image_files:
        file_path = os.path.join(input_dir, filename)
        img = Image.open(file_path)
        # Convert to RGB if necessary (in case of RGBA)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(img)

    # Save the GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # 0 means loop forever
    )
    print(f"GIF saved to {output_gif_path}")


# Example usage
if __name__ == "__main__":
    # Directory containing the PNG plots
    plot_dir = "outputs/plots/unconditional_generation/steps"
    # Output GIF path
    output_gif = "outputs/plots/unconditional_generation/steps.gif"

    # Create the GIF
    create_gif_from_plots(plot_dir, output_gif, duration=150)