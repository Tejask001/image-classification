import cv2
import numpy as np
import os

def is_blurred_advanced(image_path, laplacian_threshold=100.0, edge_threshold=0.1, frequency_threshold=0.05):
    """
    Checks if an image is blurred using a combination of Laplacian variance,
    edge detection, and frequency analysis.

    Args:
        image_path (str): Path to the image file.
        laplacian_threshold (float): Blur threshold (variance of the Laplacian).
        edge_threshold (float): Minimum fraction of pixels that should be edges.
        frequency_threshold (float): Max amplitude of the high frequency components

    Returns:
        bool: True if the image is blurred, False otherwise.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not open or read image.")

        height, width = img.shape

        # 1. Laplacian Variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian_variance = np.var(laplacian)

        # 2. Edge Detection (Sobel)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_pixels = np.count_nonzero(edge_magnitude > 0)  # Count non-zero (edge) pixels
        edge_fraction = edge_pixels / (height * width)


        # 3. Frequency Analysis (FFT)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        # Calculate amplitude of the high frequency components and compare to the rest.
        high_frequency = magnitude_spectrum[int(height / 4):int(3 * height / 4), int(width / 4):int(3 * width / 4)]
        high_freq_amplitude = np.mean(high_frequency)
        total_amplitude = np.mean(magnitude_spectrum)

        # Normalize between 0 and 1
        if total_amplitude > 0 :
            high_freq_normalized = high_freq_amplitude / total_amplitude
        else:
            high_freq_normalized = 0


        # Blur detection Logic (combining measures)
        blurred = (
            laplacian_variance < laplacian_threshold and
            edge_fraction < edge_threshold and
            high_freq_normalized < frequency_threshold
        )
        return blurred

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def detect_blur_in_folder(folder_path, laplacian_threshold=100.0, edge_threshold=0.1, frequency_threshold=0.05):
    """
    Detects blur in all images within a folder.

    Args:
        folder_path (str): Path to the folder containing images.
        laplacian_threshold (float): Blur threshold (variance of the Laplacian).
        edge_threshold (float): Minimum fraction of pixels that should be edges.
        frequency_threshold (float): Max amplitude of the high frequency components
    """

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        blurred = is_blurred_advanced(image_path, laplacian_threshold, edge_threshold, frequency_threshold)

        if blurred is True:
            print(f"{image_file}: Image is blurred.")
        elif blurred is False:
            print(f"{image_file}: Image is not blurred.")
        else:
            print(f"{image_file}: Unable to determine if image is blurred.")


# Example usage:
folder_path = "/Users/tejaskumar/Desktop/TIL/ML/image-classification-flask/test_images"  # Replace with the path to your folder
lap_thresh = 100
edge_thresh = 0.1
freq_thresh = 0.05

detect_blur_in_folder(folder_path, lap_thresh, edge_thresh, freq_thresh)