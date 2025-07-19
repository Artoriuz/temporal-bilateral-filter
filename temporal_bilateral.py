import glob
import cv2
import numpy as np

def bilateral_filter(frames, temporal_coeff=2.0, intensity_coeff=0.01):
    """
    Applies a temporal bilateral filter to a stack of grayscale frames.

    Args:
        frames (np.ndarray): Array of shape (T, H, W) with float32 values in [0, 1].
        temporal_distances (np.ndarray): Array of shape (T, 1, 1) with temporal distances to center frame.
        intensity_distances (np.ndarray): Array of shape (T, H, W) with pixelwise intensity differences to center frame.
        temporal_coeff (float): Coefficient controlling decay based on temporal distance (sigma_t).
        intensity_coeff (float): Coefficient controlling decay based on intensity distance (sigma_i).

    Returns:
        np.ndarray: Denoised 2D image (H, W) as a float32 array.
    """
    # Compute distances
    num_frames = frames.shape[0]
    center_index = num_frames // 2

    temporal_distances = np.abs(np.arange(num_frames) - center_index)
    temporal_distances = temporal_distances[:, None, None]

    intensity_distances = np.abs(frames - frames[center_index])

    # Compute bilateral weights
    temporal_weights = np.exp(-(temporal_distances ** 2) / (2 * temporal_coeff ** 2))
    intensity_weights = np.exp(-(intensity_distances ** 2) / (2 * intensity_coeff ** 2))

    # Total weights
    weights = temporal_weights * intensity_weights  # shape (T, H, W)

    # Weighted average
    weighted_sum = np.sum(weights * frames, axis=0)
    weight_total = np.sum(weights, axis=0)

    # Avoid dividing by zero
    denoised = weighted_sum / (weight_total + 1e-8)
    denoised = np.clip(denoised, 0.0, 1.0)
    return denoised

filelist = sorted(glob.glob("./frame*"))
print(filelist)
frames = []

for myFile in filelist:
    image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
    frames.append(image)
frames = np.array(frames, dtype=np.float32) / 255.0
frames = np.clip(frames, 0.0, 1.0)

denoised = bilateral_filter(frames)

denoised = np.squeeze((np.around(denoised * 255.0)).astype(np.uint8))
cv2.imwrite('./denoised.png', denoised)
