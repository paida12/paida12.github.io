import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def computeH(img1_pts, img2_pts):
    n = img1_pts.shape[0]
    
    A = np.zeros((2*n, 8))
    b = np.zeros(2*n)
    
    for i in range(n):
        x, y = img1_pts[i]
        xp, yp = img2_pts[i]
        
        A[2*i, 0] = x
        A[2*i, 1] = y
        A[2*i, 2] = 1
        A[2*i, 3] = 0
        A[2*i, 4] = 0
        A[2*i, 5] = 0
        A[2*i, 6] = -x*xp
        A[2*i, 7] = -y*xp
        b[2*i] = xp
        
        A[2*i+1, 0] = 0
        A[2*i+1, 1] = 0
        A[2*i+1, 2] = 0
        A[2*i+1, 3] = x
        A[2*i+1, 4] = y
        A[2*i+1, 5] = 1
        A[2*i+1, 6] = -x*yp
        A[2*i+1, 7] = -y*yp
        b[2*i+1] = yp
    
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]])
    
    return H

def warpImageNearestNeighbor(im, H):
    h, w = im.shape[:2]

    corners = np.array([
        [0, 0, 1],    # top left
        [w-1, 0, 1],  # top right
        [w-1, h-1, 1], # bottom right
        [0, h-1, 1]   # bottom left
    ]).T
    
    warped_corners = H @ corners
    temp = warped_corners[:2]
    warped_corners = temp / warped_corners[2]
    
    min_x, max_x = int(np.floor(warped_corners[0].min())), int(np.ceil(warped_corners[0].max()))
    min_y, max_y = int(np.floor(warped_corners[1].min())), int(np.ceil(warped_corners[1].max()))
    
    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1
    
    if len(im.shape) == 3:
        imwarped = np.zeros((out_h, out_w, im.shape[2]), dtype=im.dtype)
    else:
        imwarped = np.zeros((out_h, out_w), dtype=im.dtype)
    
    mask = np.zeros((out_h, out_w), dtype=bool)
    
    H_inv = np.linalg.inv(H)
    
    for y_out in range(out_h):
        for x_out in range(out_w):
            x_world = x_out + min_x
            y_world = y_out + min_y
            
            src_coord = H_inv @ np.array([x_world, y_world, 1])
            src_coord = src_coord[:2] / src_coord[2]

            x_src, y_src = src_coord
            
            if 0 <= x_src < w-1 and 0 <= y_src < h-1:
                x_nn = int(round(x_src))
                y_nn = int(round(y_src))
                
                x_nn = max(0, min(x_nn, w-1))
                y_nn = max(0, min(y_nn, h-1))
                
                imwarped[y_out, x_out] = im[y_nn, x_nn]
                mask[y_out, x_out] = True
    
    return imwarped, mask

def warpImageBilinear(im, H):
    h, w = im.shape[:2]
    corners = np.array([
        [0, 0, 1],    # top left
        [w-1, 0, 1],  # top right
        [w-1, h-1, 1], # bottom right
        [0, h-1, 1]   # bottom left
    ]).T
    
    warped_corners = H @ corners
    temp = warped_corners[:2]
    warped_corners = temp / warped_corners[2]

    min_x, max_x = int(np.floor(warped_corners[0].min())), int(np.ceil(warped_corners[0].max()))
    min_y, max_y = int(np.floor(warped_corners[1].min())), int(np.ceil(warped_corners[1].max()))
    
    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1
    
    if len(im.shape) == 3:
        imwarped = np.zeros((out_h, out_w, im.shape[2]), dtype=im.dtype)
    else:
        imwarped = np.zeros((out_h, out_w), dtype=im.dtype)
    
    mask = np.zeros((out_h, out_w), dtype=bool)
    
    H_inv = np.linalg.inv(H)
    
    for y_out in range(out_h):
        for x_out in range(out_w):
            x_world = x_out + min_x
            y_world = y_out + min_y
            
            src_coord = H_inv @ np.array([x_world, y_world, 1])
            src_coord = src_coord[:2] / src_coord[2]
            
            x_src, y_src = src_coord
            
            if 0 <= x_src < w-1 and 0 <= y_src < h-1:
                x1, y1 = int(np.floor(x_src)), int(np.floor(y_src))
                x2, y2 = x1 + 1, y1 + 1
                
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                dx = x_src - x1
                dy = y_src - y1
                
                if len(im.shape) == 3:
                    p11 = im[y1, x1].astype(np.float64)
                    p12 = im[y1, x2].astype(np.float64)
                    p21 = im[y2, x1].astype(np.float64)
                    p22 = im[y2, x2].astype(np.float64)
                    
                    interpolated = (p11 * (1-dx) * (1-dy) + 
                                  p12 * dx * (1-dy) + 
                                  p21 * (1-dx) * dy + 
                                  p22 * dx * dy)
                    
                    imwarped[y_out, x_out] = np.clip(interpolated, 0, 255).astype(im.dtype)
                else:
                    p11 = float(im[y1, x1])
                    p12 = float(im[y1, x2])
                    p21 = float(im[y2, x1])
                    p22 = float(im[y2, x2])
                    
                    interpolated = (p11 * (1-dx) * (1-dy) + 
                                  p12 * dx * (1-dy) + 
                                  p21 * (1-dx) * dy + 
                                  p22 * dx * dy)
                    
                    imwarped[y_out, x_out] = np.clip(interpolated, 0, 255).astype(im.dtype)
                
                mask[y_out, x_out] = True
    
    return imwarped, mask

def create_alpha_mask(shape, feather_size=50):
    h, w = shape

    alpha_mask = np.ones((h, w), dtype=np.float32)
    dist_top = np.arange(h).reshape(-1, 1)
    dist_bottom = np.arange(h-1, -1, -1).reshape(-1, 1)
    dist_left = np.arange(w).reshape(1, -1)
    dist_right = np.arange(w-1, -1, -1).reshape(1, -1)
    
    dist_to_edge = np.minimum(
        np.minimum(dist_top, dist_bottom),
        np.minimum(dist_left, dist_right)
    )
    
    alpha_mask = np.maximum(0, np.minimum(1, dist_to_edge / feather_size))
    
    return alpha_mask

def buildMosaic(images, homographies):
    all_corners = []
    
    for i, (img, H) in enumerate(zip(images, homographies)):
        h, w = img.shape[:2]
        
        corners = np.array([
            [0, 0, 1],    # top left
            [w-1, 0, 1],  # top right
            [w-1, h-1, 1], # bottom right
            [0, h-1, 1]   # bottom left
        ]).T
    
        warped_corners = H @ corners
        warped_corners = warped_corners[:2] / warped_corners[2]
        
        all_corners.append(warped_corners)
    
    all_corners_array = np.concatenate(all_corners, axis=1)
    min_x, max_x = int(np.floor(all_corners_array[0].min())), int(np.ceil(all_corners_array[0].max()))
    min_y, max_y = int(np.floor(all_corners_array[1].min())), int(np.ceil(all_corners_array[1].max()))
    
    mosaic_w = max_x - min_x + 1
    mosaic_h = max_y - min_y + 1
    num_channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    if num_channels == 1:
        mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.float64)
    else:
        mosaic = np.zeros((mosaic_h, mosaic_w, num_channels), dtype=np.float64)
     
    weight_map = np.zeros((mosaic_h, mosaic_w), dtype=np.float64)
    
    for i, (img, H) in enumerate(zip(images, homographies)):
        
        warped_img, mask = warpImageBilinear(img, H)
        
        h, w = img.shape[:2]
        corners = np.array([
            [0, 0, 1],    # top left
            [w-1, 0, 1],  # top right
            [w-1, h-1, 1], # bottom right
            [0, h-1, 1]   # bottom left
        ]).T
        
        warped_corners = H @ corners
        warped_corners = warped_corners[:2] / warped_corners[2]
        
        min_x_warped = int(np.floor(warped_corners[0].min()))
        max_x_warped = int(np.ceil(warped_corners[0].max()))
        min_y_warped = int(np.floor(warped_corners[1].min()))
        max_y_warped = int(np.ceil(warped_corners[1].max()))
        
        mosaic_x_start = max(0, min_x_warped - min_x)
        mosaic_y_start = max(0, min_y_warped - min_y)
        mosaic_x_end = min(mosaic_w, max_x_warped - min_x + 1)
        mosaic_y_end = min(mosaic_h, max_y_warped - min_y + 1)
        
        h_warped, w_warped = warped_img.shape[:2]
        warp_x_start = 0
        warp_y_start = 0
        warp_x_end = w_warped
        warp_y_end = h_warped
        
        warp_x_end = min(warp_x_end, w_warped)
        warp_y_end = min(warp_y_end, h_warped)
        
        mosaic_region_h = mosaic_y_end - mosaic_y_start
        mosaic_region_w = mosaic_x_end - mosaic_x_start
        
        if mosaic_region_h > 0 and mosaic_region_w > 0:
            warped_region = warped_img[warp_y_start:warp_y_end, warp_x_start:warp_x_end]
            mask_region = mask[warp_y_start:warp_y_end, warp_x_start:warp_x_end]
            
            alpha_mask = create_alpha_mask((mosaic_region_h, mosaic_region_w))
            
            alpha_mask = alpha_mask * mask_region
            mosaic[mosaic_y_start:mosaic_y_end, mosaic_x_start:mosaic_x_end] += warped_region * alpha_mask[:, :, np.newaxis] if num_channels > 1 else warped_region * alpha_mask
            
            weight_map[mosaic_y_start:mosaic_y_end, mosaic_x_start:mosaic_x_end] += alpha_mask
    
    weight_map = np.maximum(weight_map, 1e-10)
    
    if num_channels > 1:
        for c in range(num_channels):
            mosaic[:, :, c] = mosaic[:, :, c] / weight_map
    else:
        mosaic = mosaic / weight_map
    
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    return mosaic
