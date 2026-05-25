import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """Mô phỏng thêm nhiễu Gaussian vào ảnh ảnh gốc"""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image.astype(np.float64) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """Mô phỏng thêm nhiễu muối tiêu (nhiễu xung)"""
    noisy_image = image.copy()
    # Thêm hạt muối (màu trắng - 255)
    num_salt = np.ceil(salt_prob * image.size / (image.shape[2] if len(image.shape)==3 else 1))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Thêm hạt tiêu (màu đen - 0)
    num_pepper = np.ceil(pepper_prob * image.size / (image.shape[2] if len(image.shape)==3 else 1))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

def calculate_mse(original, restored):
    """Tính toán Mean Squared Error (MSE) theo công thức tài liệu"""
    orig_f = original.astype(np.float64)
    rest_f = restored.astype(np.float64)
    mse = np.mean((orig_f - rest_f) ** 2)
    return mse

def calculate_snr(original, restored):
    """Tính toán tỷ số Signal-to-Noise Ratio (SNR) theo công thức tài liệu"""
    orig_f = original.astype(np.float64)
    rest_f = restored.astype(np.float64)
    numerator = np.sum(rest_f ** 2)
    denominator = np.sum((orig_f - rest_f) ** 2)
    if denominator == 0:
        return float('inf')
    return numerator / denominator

def add_uniform_noise(image, a=-20, b=20):
    """Mô phỏng thêm nhiễu đồng đều (Uniform Noise)"""
    noise = np.random.uniform(a, b, image.shape)
    noisy_image = image.astype(np.float64) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_erlang_noise(image, a=10.0, b=2):
    """
    Mô phỏng thêm nhiễu Erlang (Gamma).
    a: Tham số tỉ lệ (scale), b: Số nguyên thể hiện hình dạng (shape - bậc).
    """
    noise = np.random.gamma(b, a, image.shape)
    # Trừ đi giá trị kỳ vọng (b * a) để giữ cân bằng sáng cho ảnh
    noise = noise - (b * a)
    noisy_image = image.astype(np.float64) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_rayleigh_noise(image, a=0.0, scale=15.0):
    """
    Mô phỏng nhiễu Rayleigh.
    scale: Độ lệch chuẩn Rayleigh, a: độ dịch vị trí (offset).
    """
    noise = np.random.rayleigh(scale, image.shape) + a
    # Trừ đi kỳ vọng lý thuyết để giữ cân bằng sáng
    mean_val = a + scale * np.sqrt(np.pi / 2)
    noise = noise - mean_val
    noisy_image = image.astype(np.float64) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_periodic_noise(image, amplitude=20.0, u0=10.0, v0=10.0):
    """
    Mô phỏng thêm nhiễu tuần hoàn hình sin.
    amplitude: Biên độ nhiễu.
    u0, v0: Tần số nhiễu theo các chiều.
    """
    H, W = image.shape[:2]
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    # Sóng sin nhiễu tuần hoàn
    noise_wave = amplitude * np.sin(2 * np.pi * (u0 * Y / H + v0 * X / W))
    
    # Đồng bộ hóa chiều ma trận nếu là ảnh đa kênh RGB
    if len(image.shape) == 3:
        noise_wave = np.expand_dims(noise_wave, axis=-1)
        
    noisy_image = image.astype(np.float64) + noise_wave
    return np.clip(noisy_image, 0, 255).astype(np.uint8)