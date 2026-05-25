import numpy as np
import cv2

def get_fft_spectrum(image_gray):
    """
    Tính toán phổ biên độ Fourier (đã shift và log-transform) để hiển thị trực quan.
    """
    f = np.fft.fft2(image_gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    # Chuẩn hóa về [0, 255] để hiển thị
    mag_max = np.max(magnitude_spectrum)
    if mag_max > 0:
        magnitude_spectrum = (magnitude_spectrum / mag_max) * 255
    return magnitude_spectrum.astype(np.uint8), fshift

def reconstruct_from_fft(fshift):
    """
    Khôi phục ảnh từ ma trận phổ tần số (đã shift).
    """
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)

def compute_distances(shape):
    """
    Tính ma trận khoảng cách D(u, v) từ tâm tần số.
    """
    H, W = shape
    u = np.arange(H)
    v = np.arange(W)
    # Chuyển dịch tọa độ về tâm
    u = u - H / 2
    v = v - W / 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    return D

def bandreject_filter_mask(shape, D0, W, filter_type="Ideal", n=1):
    """
    Tạo mặt nạ bộ lọc chặn dải (Bandreject Filter) H(u, v)
    """
    D = compute_distances(shape)
    # Tránh chia cho 0
    D[D == 0] = 1e-5
    
    if filter_type == "Ideal":
        mask = np.ones(shape, dtype=np.float64)
        mask[(D >= D0 - W/2) & (D <= D0 + W/2)] = 0
        return mask
    elif filter_type == "Butterworth":
        # H = 1 / (1 + [D*W / (D^2 - D0^2)]^(2n))
        num = D * W
        den = D**2 - D0**2
        den[den == 0] = 1e-5
        mask = 1.0 / (1.0 + np.power(num / den, 2 * n))
        return mask
    elif filter_type == "Gaussian":
        # H = 1 - exp(-0.5 * [(D^2 - D0^2) / (D*W)]^2)
        exponent = (D**2 - D0**2) / (D * W)
        mask = 1.0 - np.exp(-0.5 * np.power(exponent, 2))
        return mask
    return np.ones(shape, dtype=np.float64)

def notch_filter_mask(shape, D0, uk, vk, filter_type="Ideal", n=1, mode="Reject"):
    """
    Tạo mặt nạ bộ lọc Notch (Notch Filter) H(u, v) đối xứng qua tâm.
    uk, vk là độ lệch tần số so với tâm.
    """
    H, W = shape
    u = np.arange(H) - H / 2
    v = np.arange(W) - W / 2
    V, U = np.meshgrid(v, u)
    
    # Khoảng cách đến tâm nhiễu 1 (uk, vk) và tâm nhiễu 2 (-uk, -vk)
    D1 = np.sqrt((U - uk)**2 + (V - vk)**2)
    D2 = np.sqrt((U + uk)**2 + (V + vk)**2)
    
    D1[D1 == 0] = 1e-5
    D2[D2 == 0] = 1e-5
    
    if filter_type == "Ideal":
        mask = np.ones(shape, dtype=np.float64)
        mask[D1 <= D0] = 0
        mask[D2 <= D0] = 0
    elif filter_type == "Butterworth":
        # H = 1 / (1 + [D0^2 / (D1 * D2)]^n)
        mask = 1.0 / (1.0 + np.power(D0**2 / (D1 * D2), n))
    elif filter_type == "Gaussian":
        # H = 1 - exp(-0.5 * [D1 * D2 / D0^2])
        mask = 1.0 - np.exp(-0.5 * (D1 * D2) / (D0**2))
    else:
        mask = np.ones(shape, dtype=np.float64)
        
    if mode == "Pass":
        mask = 1.0 - mask
        
    return mask

def wiener_filter_restoration(image_gray, D0=30, K=0.01, noise_sigma=10):
    """
    Mô phỏng ảnh bị nhòe (Blur) bằng bộ lọc Gaussian Lowpass H(u, v),
    thêm nhiễu Gaussian, sau đó khôi phục lại bằng bộ lọc Wiener.
    """
    H, W = image_gray.shape
    f = np.fft.fft2(image_gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    
    # 1. Tạo hàm suy hao Gaussian Lowpass H(u, v) đại diện cho độ nhòe
    D = compute_distances((H, W))
    H_blur = np.exp(-(D**2) / (2 * (D0**2)))
    
    # 2. Làm nhòe ảnh trong miền tần số
    fshift_blurred = fshift * H_blur
    blurred_img = reconstruct_from_fft(fshift_blurred)
    
    # 3. Thêm nhiễu Gaussian vào ảnh nhòe
    noise = np.random.normal(0, noise_sigma, (H, W))
    noisy_blurred_img = np.clip(blurred_img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    # Lấy FFT của ảnh nhòe nhiễu
    g = np.fft.fft2(noisy_blurred_img.astype(np.float64))
    gshift = np.fft.fftshift(g)
    
    # 4. Áp dụng bộ lọc Wiener
    # F_hat = [ H / (H^2 + K) ] * G
    # Đối với H thực
    wiener_filter = H_blur / (H_blur**2 + K)
    fshift_restored = gshift * wiener_filter
    restored_img = reconstruct_from_fft(fshift_restored)
    
    # Trả về: Ảnh nhòe nhiễu, Ảnh phục hồi, Mặt nạ bộ lọc, Phổ tần số ảnh phục hồi
    # Để hiển thị phổ tần số ảnh phục hồi
    restored_spectrum = 20 * np.log(np.abs(fshift_restored) + 1)
    mag_max = np.max(restored_spectrum)
    if mag_max > 0:
        restored_spectrum = (restored_spectrum / mag_max) * 255
        
    return noisy_blurred_img, restored_img, (H_blur * 255).astype(np.uint8), restored_spectrum.astype(np.uint8)
