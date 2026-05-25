import numpy as np
import cv2

def arithmetic_mean_filter(image, kernel_size=3):
    """Bộ lọc trung bình số học loại bỏ biến đổi cục bộ (Nhiễu Gaussian/Đồng đều)"""
    return cv2.blur(image, (kernel_size, kernel_size))

def geometric_mean_filter(image, kernel_size=3):
    """Bộ lọc trung bình hình học giữ chi tiết tốt hơn trung bình số học"""
    # Sử dụng miền Logarithm để tích các điểm ảnh lân cận tránh hiện tượng tràn số nguyên
    image_float = image.astype(np.float64) + 1e-5
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size * kernel_size)
    
    if len(image.shape) == 2:
        log_img = np.log(image_float)
        sum_log = cv2.filter2D(log_img, -1, kernel)
        return np.clip(np.exp(sum_log), 0, 255).astype(np.uint8)
    else:
        channels = []
        for i in range(image.shape[2]):
            log_ch = np.log(image_float[:, :, i])
            sum_log = cv2.filter2D(log_ch, -1, kernel)
            channels.append(np.exp(sum_log))
        return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)

def median_filter(image, kernel_size=3):
    """Bộ lọc thống kê thứ tự trung vị đặc trị loại bỏ nhiễu Muối Tiêu"""
    return cv2.medianBlur(image, kernel_size)

def max_filter(image, kernel_size=3):
    """Bộ lọc cực đại tìm điểm sáng nhất lân cận, trị nhiễu hạt tiêu"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel)

def min_filter(image, kernel_size=3):
    """Bộ lọc cực tiểu tìm điểm tối nhất lân cận, trị nhiễu hạt muối"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(image, kernel)

def harmonic_mean_filter(image, kernel_size=3):
    """Bộ lọc trung bình điều hòa: loại bỏ nhiễu muối tốt, không lọc được nhiễu tiêu"""
    image_float = image.astype(np.float64) + 1e-5
    inv_img = 1.0 / image_float
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    
    if len(image.shape) == 2:
        sum_inv = cv2.filter2D(inv_img, -1, kernel)
        res = (kernel_size * kernel_size) / sum_inv
        return np.clip(res, 0, 255).astype(np.uint8)
    else:
        channels = []
        for i in range(image.shape[2]):
            sum_inv = cv2.filter2D(inv_img[:, :, i], -1, kernel)
            res = (kernel_size * kernel_size) / sum_inv
            channels.append(res)
        return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)

def contraharmonic_mean_filter(image, kernel_size=3, Q=1.5):
    """Bộ lọc trung bình nghịch đảo điều hòa (Contraharmonic Mean Filter)"""
    image_float = image.astype(np.float64) + 1e-5
    
    # Để tránh tràn số hoặc số quá lớn/nhỏ khi lũy thừa Q lớn
    # Cắt xén Q hợp lý
    num = np.power(image_float, Q + 1)
    den = np.power(image_float, Q)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    
    if len(image.shape) == 2:
        sum_num = cv2.filter2D(num, -1, kernel)
        sum_den = cv2.filter2D(den, -1, kernel)
        sum_den[sum_den == 0] = 1e-5
        res = sum_num / sum_den
        return np.clip(res, 0, 255).astype(np.uint8)
    else:
        channels = []
        for i in range(image.shape[2]):
            sum_num = cv2.filter2D(num[:, :, i], -1, kernel)
            sum_den = cv2.filter2D(den[:, :, i], -1, kernel)
            sum_den[sum_den == 0] = 1e-5
            channels.append(sum_num / sum_den)
        return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)

def midpoint_filter(image, kernel_size=3):
    """Bộ lọc điểm giữa (Midpoint Filter)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if len(image.shape) == 2:
        img_max = cv2.dilate(image, kernel)
        img_min = cv2.erode(image, kernel)
        res = (img_max.astype(np.float64) + img_min.astype(np.float64)) / 2.0
        return np.clip(res, 0, 255).astype(np.uint8)
    else:
        channels = []
        for i in range(image.shape[2]):
            img_max = cv2.dilate(image[:, :, i], kernel)
            img_min = cv2.erode(image[:, :, i], kernel)
            res = (img_max.astype(np.float64) + img_min.astype(np.float64)) / 2.0
            channels.append(res)
        return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)

def alpha_trimmed_mean_filter(image, kernel_size=3, d=2):
    """Bộ lọc trung vị điều chỉnh (Alpha-trimmed Mean Filter)"""
    if d < 0 or d >= kernel_size * kernel_size:
        d = 0
        
    pad = kernel_size // 2
    if len(image.shape) == 2:
        padded = np.pad(image, pad, mode='edge')
        H, W = image.shape
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        flat_windows = windows.reshape(H, W, -1)
        sorted_windows = np.sort(flat_windows, axis=-1)
        trim_low = d // 2
        trim_high = (kernel_size * kernel_size) - (d - trim_low)
        trimmed = sorted_windows[:, :, trim_low:trim_high]
        res = np.mean(trimmed, axis=-1)
        return np.clip(res, 0, 255).astype(np.uint8)
    else:
        channels = []
        for i in range(image.shape[2]):
            channels.append(alpha_trimmed_mean_filter(image[:, :, i], kernel_size, d))
        return np.stack(channels, axis=2)