import cv2
import numpy as np

def image_negative(image):
    """Tạo ảnh âm bản: s = L - 1 - r"""
    return 255 - image

def log_transformation(image, c=1.0):
    """Biến đổi Logarithm mở rộng vùng tối: s = c * log(1 + r)"""
    image_float = image.astype(float) / 255.0
    log_img = c * np.log(1 + image_float)
    # Chuẩn hóa ma trận điểm ảnh về dải màu hợp lệ
    log_img = log_img / np.max(log_img) if np.max(log_img) > 0 else log_img
    return np.clip(log_img * 255, 0, 255).astype(np.uint8)

def gamma_transformation(image, c=1.0, gamma=1.0):
    """Biến đổi lũy thừa làm sáng/tối ảnh linh hoạt: s = c * r^γ"""
    image_float = image.astype(float) / 255.0
    gamma_img = c * np.power(image_float, gamma)
    return np.clip(gamma_img * 255, 0, 255).astype(np.uint8)

def histogram_equalization(image):
    """Cân bằng lược đồ xám nâng cao độ tương phản tổng thể"""
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        # Đối với ảnh màu RGB, chuyển sang hệ YCrCb để cân bằng kênh độ sáng (Y)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def sharpening_laplacian(image):
    """Làm sắc nét ảnh sử dụng đạo hàm bậc 2 toán tử Laplacian"""
    # Sử dụng ma trận nhân (kernel) trung tâm dương kết hợp cộng trực tiếp biên vào ảnh gốc
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def piecewise_linear_transformation(image, r1, s1, r2, s2):
    """Biến đổi tuyến tính từng đoạn (Contrast Stretching)"""
    img_float = image.astype(np.float64)
    res = np.zeros_like(img_float)
    
    # Đoạn 1: 0 <= r < r1
    mask1 = img_float < r1
    if r1 > 0:
        res[mask1] = (s1 / r1) * img_float[mask1]
    else:
        res[mask1] = 0
        
    # Đoạn 2: r1 <= r <= r2
    mask2 = (img_float >= r1) & (img_float <= r2)
    if r2 > r1:
        res[mask2] = ((s2 - s1) / (r2 - r1)) * (img_float[mask2] - r1) + s1
    else:
        res[mask2] = s1
        
    # Đoạn 3: r2 < r <= 255
    mask3 = img_float > r2
    if r2 < 255:
        res[mask3] = ((255 - s2) / (255 - r2)) * (img_float[mask3] - r2) + s2
    else:
        res[mask3] = s2
        
    return np.clip(res, 0, 255).astype(np.uint8)

def gray_level_slicing(image, low, high, value=255, preserve_background=True):
    """Cắt ngưỡng độ xám (Gray-level Slicing)"""
    res = image.copy()
    mask = (res >= low) & (res <= high)
    
    if preserve_background:
        res[mask] = value
    else:
        res[mask] = value
        res[~mask] = 0
        
    return res

def bit_plane_slicing(image, bit_idx):
    """Trích xuất mặt phẳng bit thứ bit_idx (0 đến 7)"""
    bit_img = (image >> bit_idx) & 1
    return (bit_img * 255).astype(np.uint8)

def reconstruct_bit_planes(image, start_bit):
    """Tái tạo ảnh bằng cách chỉ giữ lại các mặt phẳng bit từ start_bit đến 7"""
    mask = 0
    for i in range(start_bit, 8):
        mask |= (1 << i)
    return (image & mask).astype(np.uint8)

def gaussian_smoothing_filter(image, kernel_size=3, sigma=0.0):
    """Bộ lọc Gaussian làm mượt ảnh"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def box_filter(image, width=3, height=3, normalize=True):
    """Bộ lọc Box filter tùy biến kích thước mxn, có hoặc không chuẩn hóa"""
    return cv2.boxFilter(image, -1, (width, height), normalize=normalize)

def first_derivative_sharpening(image, operator_type="Sobel", mode="Chỉ hiển thị biên (Edge magnitude)", weight=0.5):
    """Làm sắc nét hoặc trích xuất biên bằng đạo hàm bậc nhất: Sobel, Prewitt, Roberts"""
    img_float = image.astype(np.float32)
    
    if operator_type == "Sobel":
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    elif operator_type == "Prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    elif operator_type == "Roberts":
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown operator: {operator_type}")
        
    gx = cv2.filter2D(img_float, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(img_float, cv2.CV_32F, kernel_y)
    grad = np.sqrt(gx**2 + gy**2)
        
    if mode == "Chỉ hiển thị biên (Edge magnitude)":
        return np.clip(grad, 0, 255).astype(np.uint8)
    else:
        # Làm sắc nét: s = f + weight * grad
        sharpened = img_float + weight * grad
        return np.clip(sharpened, 0, 255).astype(np.uint8)