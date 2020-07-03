import cv2
import numpy as np

def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def slice_mask_with_border(mask_shape, slice_size, border_size, mask_arr, margin=0):
    mask = np.zeros(mask_shape)

    h, w = mask_shape
    rows = (h - 2 * border_size) // slice_size
    cols = (w - 2 * border_size) // slice_size
    
    for i in range(rows):
            for j in range(cols):
                mask[
                    i*slice_size+margin+border_size:(i+1)*slice_size-margin+border_size,
                    j*slice_size+margin+border_size:(j+1)*slice_size-margin+border_size
                ] = 1 if mask_arr[i*cols+j] > 0 else 0
                
    return mask

def slice_plot(img_path, slice_size, border_size, mask_arr, mask_color):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = slice_mask_with_border(
        (img.shape[0], img.shape[1]),
        slice_size,
        border_size,
        mask_arr
    )
    masked_img = apply_mask(img, mask, mask_color)
    
    return masked_img
    
def merge_patch(patches, cols, rows, width, height):
    img = np.zeros((rows*height, cols*width))
    
    for i in range(rows):
        for j in range(cols):
            img[i*height:(i+1)*height, j*width:(j+1)*width] = patches[i*cols+j]
            
    return img