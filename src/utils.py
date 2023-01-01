def normalize_img_intensity_range(img):
    min_val, max_val = np.min(img), np.max(img)
    range = max_val - min_val
    return (img - min_val) / range