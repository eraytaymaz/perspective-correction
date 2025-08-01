#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim


# In[2]:


def load_image(image_path, target_size=(1024,1024)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


# In[3]:


def preprocess_for_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (11, 11), 0) 
    return gaussian


# In[4]:


def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape
    img_diagonal = np.sqrt(height**2 + width**2)
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
    H = np.zeros((len(rhos), len(thetas)), dtype=np.float32)
    y_idxs, x_idxs = np.nonzero(img)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for x, y in zip(x_idxs, y_idxs):
        rho_vals = (x * cos_t + y * sin_t) + img_diagonal  # keep floating point precision
        rho_idxs = np.round(rho_vals / rho_resolution).astype(int)
        H[rho_idxs, np.arange(len(thetas))] += 1  # vectorized update
    return H, rhos, thetas


# In[5]:


def hough_peaks(H, num_peaks, threshold=10, nhood_size=7):
    indices = []
    H1 = np.copy(H)
    
    for i in range(num_peaks):
        idx = np.argmax(H1)  # flattened argmax
        peak_value = H1.flat[idx]
        
        # Break if the maximum value is below the threshold.
        if peak_value < threshold:
            break
        
        H1_idx = np.unravel_index(idx, H1.shape)
        indices.append(H1_idx)
        idx_y, idx_x = H1_idx
        
        half_n = nhood_size // 2
        min_x = max(idx_x - half_n, 0)
        max_x = min(idx_x + half_n + 1, H.shape[1])
        min_y = max(idx_y - half_n, 0)
        max_y = min(idx_y + half_n + 1, H.shape[0])
        
        # Suppress the neighborhood in H1 using slicing
        H1[min_y:max_y, min_x:max_x] = 0
        
        H[min_y, min_x:max_x] = 255
        H[max_y - 1, min_x:max_x] = 255
        H[min_y:max_y, min_x] = 255
        H[min_y:max_y, max_x - 1] = 255

    return indices, H


# In[6]:


def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    #Plots the Hough accumulator using matplotlib.
    plt.figure(figsize=(10, 10))
    plt.imshow(H.T, cmap='jet')
    plt.xlabel('Rho Direction')
    plt.ylabel('Theta Direction')
    plt.title(plot_title)
    plt.tight_layout()
    plt.show()


# In[7]:


def hough_lines_draw(img, indices, rhos, thetas):
    #Draws lines corresponding to the provided indices on the image.
    for idx in indices:
        rho = rhos[idx[0]]
        theta = thetas[idx[1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend the line to cover the image dimensions
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


# In[8]:


def ransac_line_detection(edge_image, hough_lines, num_iterations = 100, distance_threshold=10, min_inliers=100, num_lines=10, min_rho_diff=100, min_theta_diff=np.deg2rad(25)):
    best_lines = []
    edge_points = np.column_stack(np.where(edge_image > 0))
    
    if len(edge_points) == 0:
        return []
    
    edge_points = edge_points.astype(np.float32)
    hough_lines = np.array(hough_lines)
    for _ in range(num_iterations):
        if len(best_lines) >= num_lines:
            break
        
        # Randomly select a candidate line from the Hough lines
        rho, theta = hough_lines[random.randint(0, len(hough_lines) - 1)]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        # Compute distances of all edge points to the candidate line
        distances = np.abs(rho - (edge_points[:, 1] * cos_theta + edge_points[:, 0] * sin_theta))
        inliers = edge_points[distances < distance_threshold]
    
        # Check if the candidate line has enough inliers
        if len(inliers) >= min_inliers:
            similar = False
            for r, t in best_lines:
                if abs(rho - r) < min_rho_diff and abs(theta - t) < min_theta_diff:
                    similar = True
                    break
            if not similar:
                best_lines.append((rho, theta))
    
    return best_lines


# In[9]:


def plot_detected_lines(image, best_lines):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    height, width = image.shape
    
    for rho, theta in best_lines:
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x1 = int(rho * cos_theta + 1000 * (-sin_theta))
        y1 = int(rho * sin_theta + 1000 * (cos_theta))
        x2 = int(rho * cos_theta - 1000 * (-sin_theta))
        y2 = int(rho * sin_theta - 1000 * (cos_theta))
        plt.plot([x1, x2], [y1, y2], 'r')
    
    plt.xlim([0, width])
    plt.ylim([height, 0])
    plt.title('Detected {} Lines using RANSAC'.format(len(best_lines)))
    plt.show()


# In[10]:


def detect_quadrilateral(image, lines, min_area_ratio=0.01, max_area_ratio=0.99):
    intersections = []
    height, width = image.shape[:2]
    img_area = height * width
    # Compute intersections for every pair of lines
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]
            
            # Avoid nearly parallel lines
            if abs(theta1 - theta2) < 1e-3:
                continue
            
            #x*cos(theta1) + y*sin(theta1) = rho1
            #x*cos(theta2) + y*sin(theta2) = rho2
            A = np.array([[np.cos(theta1), np.sin(theta1)],
                          [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            try:
                x0, y0 = np.linalg.solve(A, b)
                # Accept intersections within image boundaries
                if 0 <= x0 <= width and 0 <= y0 <= height:
                    intersections.append([x0, y0])
            except np.linalg.LinAlgError:
                continue

    if not intersections:
        return None

    intersections = np.array(intersections, dtype=np.float32)
    hull = cv2.convexHull(intersections)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) == 4:
        quad = approx.reshape(4, 2)
        quad_area = cv2.contourArea(quad)
        if quad_area < min_area_ratio * img_area or quad_area > max_area_ratio * img_area:
            return None
        return quad
    else:
        return None


# In[11]:


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # Top-left has the smallest sum, bottom-right the largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Compute the difference (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


# In[4]:


def compute_homography(src, dst):
    A = []
    for i in range(4):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H


# In[5]:


def bilinear_interpolate(im, x, y):
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    if im.ndim == 2:
        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]
    else:
        Ia = im[y0, x0, :]
        Ib = im[y1, x0, :]
        Ic = im[y0, x1, :]
        Id = im[y1, x1, :]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    if im.ndim == 2:
        return wa * Ia + wb * Ib + wc * Ic + wd * Id
    else:
        return wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id


# In[6]:


def warp_perspective(image, H, dsize):
    width, height = dsize
    H_inv = np.linalg.inv(H)
    
    # Create grid of (x, y) coordinates for destination image.
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(xx)
    dest_coords = np.stack([xx, yy, ones], axis=-1)  # shape: (height, width, 3)
    dest_coords_flat = dest_coords.reshape(-1, 3).T  # shape: (3, N)
    
    # Map destination pixels back to source coordinates
    src_coords_flat = H_inv @ dest_coords_flat
    src_coords_flat /= src_coords_flat[2, :]
    src_x = src_coords_flat[0, :].reshape(height, width)
    src_y = src_coords_flat[1, :].reshape(height, width)
    
    warped = bilinear_interpolate(image, src_x, src_y)
    return warped


# In[12]:


def perspective_correction(image, quad):
    
    rect = order_points(quad)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    H = compute_homography(rect, dst)
    warped = warp_perspective(image, H, (maxWidth, maxHeight))
    return warped


# In[31]:


def process_image(category, index, show = False):
    try:
        # Load the image
        distorted_path = f'./WarpDoc/distorted/{category}/{index}.jpg'
        shapes = load_image(distorted_path, target_size=(1024, 1024))
        # Preprocess for edge detection
        processed_for_edges = preprocess_for_edges(shapes)

        # Use Canny edge detection
        canny_edges = cv2.Canny(processed_for_edges, 50, 150)

        # Compute Hough accumulator for the edge image
        H, rhos, thetas = hough_lines_acc(canny_edges)

        # Find peaks
        indices, H_highlighted = hough_peaks(H, 25, threshold=100, nhood_size=25)

        # Draw detected lines from the Hough transform
        shapes_with_lines = shapes.copy()
        hough_lines = [(rhos[idx[0]], thetas[idx[1]]) for idx in indices]
        hough_lines_draw(shapes_with_lines, indices, rhos, thetas)
        # Line detection using RANSAC
        best_lines = ransac_line_detection(canny_edges, hough_lines)


        #Detect quadrilateral
        quad = detect_quadrilateral(shapes, best_lines)
        if quad is not None:
            # Draw the detected quadrilateral on the image
            quad_img = shapes.copy()
            quad_points = quad.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(quad_img, [quad_points], isClosed=True, color=(0, 0, 255), thickness=2)
            warped = perspective_correction(shapes, quad)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            ground_truth_path = f'./WarpDoc/digital/{category}/{index}.jpg'
            ground_truth = load_image(ground_truth_path, target_size=(1024,1024))
            ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

            # Resize warped_gray if dimensions do not match ground_truth_gray
            if warped_gray.shape != ground_truth_gray.shape:
                warped_gray = cv2.resize(warped_gray, (ground_truth_gray.shape[1], ground_truth_gray.shape[0]))

            ssim_index, ssim_image = ssim(ground_truth_gray, warped_gray, full=True, data_range=255)
            print(f"SSIM for image {index} in category {category}: {ssim_index}")
            if show: show_images(canny_edges, H_highlighted, shapes_with_lines, processed_for_edges, best_lines, quad_img, warped_gray, ground_truth_gray)
            return ssim_index
        else:
            print(f"Perspective correction for image {index} skipped due to invalid quadrilateral.")
            return None
    except Exception as e:
        print(f"Error processing image {index} in category {category}: {e}")
        return None


# In[32]:


def show_images(canny_edges, H_highlighted, shapes_with_lines, processed_for_edges, best_lines, quad_img, warped_gray, ground_truth_gray):
    plt.figure(figsize=(10, 10))
    plt.imshow(canny_edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis("off")
    plt.show()
    plot_hough_acc(H_highlighted, 'Hough Accumulator with Highlighted Peaks')
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(shapes_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Major Lines: Manual Hough Transform')
    plt.axis('off')
    plt.show()
    plot_detected_lines(processed_for_edges, best_lines)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(quad_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Quadrilateral')
    plt.axis('off')
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(warped_gray, cv2.COLOR_BGR2RGB))
    plt.title("Perspective Corrected Image")
    plt.axis("off")
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(ground_truth_gray, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth Image")
    plt.axis("off")
    plt.show()


# In[53]:


categories = ["curved", "fold", "incomplete", "perspective", "random", "rotate"]
scores = {
    "curved": [], 
    "fold": [],
    "incomplete": [], 
    "perspective": [], 
    "random": [], 
    "rotate": [],
}


# In[54]:


for category in categories:
    for i in range(0, 100):
        index = f"{i:04d}"
        ssim_index = process_image(category, index)
        if ssim_index is not None: scores[category].append(ssim_index)


# In[55]:


for category in categories:
    print(f"Average SSIM for category {category}\t: {(sum(sorted(scores[category], reverse=True)[:50]))/min(50, len(scores[category]))}")


# In[33]:


process_image("curved", "0002", show = True)


# In[34]:


process_image("curved", "0096", show = True)


# In[35]:


process_image("fold", "0032", show = True)


# In[38]:


process_image("fold", "0061", show = True)


# In[39]:


process_image("incomplete", "0069", show = True)


# In[42]:


process_image("incomplete", "0088", show = True)


# In[43]:


process_image("perspective", "0049", show = True)


# In[44]:


process_image("perspective", "0058", show = True)


# In[45]:


process_image("random", "0061", show = True)


# In[47]:


process_image("random", "0028", show = True)


# In[48]:


process_image("rotate", "0060", show = True)


# In[51]:


process_image("rotate", "0077", show = True)


# In[ ]:




