import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import mã lõi thuật toán
from algorithms.enhancement import (
    image_negative, log_transformation, gamma_transformation, 
    histogram_equalization, sharpening_laplacian,
    piecewise_linear_transformation, gray_level_slicing,
    bit_plane_slicing, reconstruct_bit_planes,
    gaussian_smoothing_filter, box_filter,
    first_derivative_sharpening
)
from algorithms.restoration import (
    arithmetic_mean_filter, geometric_mean_filter, 
    median_filter, max_filter, min_filter,
    harmonic_mean_filter, contraharmonic_mean_filter,
    midpoint_filter, alpha_trimmed_mean_filter
)
from algorithms.frequency import (
    get_fft_spectrum, reconstruct_from_fft,
    bandreject_filter_mask, notch_filter_mask,
    wiener_filter_restoration
)
from utils import (
    add_gaussian_noise, add_salt_and_pepper_noise, 
    calculate_mse, calculate_snr,
    add_uniform_noise, add_erlang_noise,
    add_rayleigh_noise, add_periodic_noise
)

# Cấu hình Layout rộng cho giao diện Streamlit
st.set_page_config(layout="wide", page_title="Image Processing Application")
st.title("🖥️ Ứng Dụng Tăng Cường & Phục Hồi Ảnh Số")
st.write("Demo tích hợp hệ thống bộ lọc không gian phục vụ học tập môn Xử lý ảnh.")

# Tạo thanh bên để Upload hình ảnh đầu vào
st.sidebar.header("📁 Dữ Liệu Đầu Vào")
uploaded_file = st.sidebar.file_uploader("Tải lên hình ảnh xử lý (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh gốc bằng PIL sang mảng đa chiều NumPy dạng RGB chuẩn
    image_pil = Image.open(uploaded_file)
    original_img = np.array(image_pil)
    
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    elif original_img.shape[2] == 4:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGBA2RGB)

    # Chọn chế độ phân hệ bài toán xử lý chính
    mode = st.sidebar.selectbox("Chọn Phân Hệ Chức Năng", [
        "Tăng cường ảnh (Chapter 3)", 
        "Phục hồi ảnh (Chapter 5)",
        "Xử lý miền tần số (Chapter 4 & 5)"
    ])

    if mode == "Tăng cường ảnh (Chapter 3)":
        st.header("✨ Phân Hệ: Tăng Cường Ảnh (Spatial Domain)")
        
        algo = st.selectbox("Chọn thuật toán biến đổi hình thái", [
            "Ảnh âm bản (Image Negative)",
            "Biến đổi độ sáng Logarithm",
            "Biến đổi phân rã Lũy thừa (Gamma)",
            "Biến đổi tuyến tính từng đoạn (Piecewise Linear)",
            "Cắt ngưỡng độ xám (Gray-level Slicing)",
            "Trích xuất mặt phẳng bit (Bit-plane Slicing)",
            "Cân bằng Lược đồ xám (Histogram Equalization)",
            "Bộ lọc Gaussian làm mượt (Gaussian Blur)",
            "Bộ lọc Box filter tùy biến",
            "Làm sắc nét ảnh biên (Bộ lọc Laplacian)",
            "Làm sắc nét đạo hàm bậc nhất (Sobel, Prewitt, Roberts)"
        ])
        
        result_img = original_img.copy()
        
        # Cấu hình thanh trượt tham số động tương thích từng hàm toán học
        if algo == "Ảnh âm bản (Image Negative)":
            result_img = image_negative(original_img)
        elif algo == "Biến đổi độ sáng Logarithm":
            c = st.slider("Hệ số mở rộng thang độ sáng (c)", 0.5, 3.0, 1.0, 0.1)
            result_img = log_transformation(original_img, c=c)
        elif algo == "Biến đổi phân rã Lũy thừa (Gamma)":
            c = st.slider("Hệ số biến đổi tỉ lệ (c)", 0.5, 2.0, 1.0, 0.1)
            gamma = st.slider("Hệ số hiệu chỉnh Gamma (γ)", 0.1, 4.0, 1.0, 0.1)
            result_img = gamma_transformation(original_img, c=c, gamma=gamma)
        elif algo == "Biến đổi tuyến tính từng đoạn (Piecewise Linear)":
            st.write("🔧 **Cấu hình điểm gãy tuyến tính (r1, s1) và (r2, s2)**")
            r1 = st.slider("Điểm r1", 0, 255, 80)
            s1 = st.slider("Điểm s1", 0, 255, 30)
            r2 = st.slider("Điểm r2", 0, 255, 180)
            s2 = st.slider("Điểm s2", 0, 255, 220)
            
            # Đảm bảo r1 <= r2 trong logic xử lý
            if r1 > r2:
                r1, r2 = r2, r1
                st.warning("⚠️ Đã tự động đổi chỗ r1 và r2 để đảm bảo r1 <= r2.")
            
            result_img = piecewise_linear_transformation(original_img, r1=r1, s1=s1, r2=r2, s2=s2)
        elif algo == "Cắt ngưỡng độ xám (Gray-level Slicing)":
            st.write("🔧 **Cấu hình ngưỡng cắt và giá trị thay thế**")
            low, high = st.slider("Chọn khoảng giá trị cắt ngưỡng [low, high]", 0, 255, (100, 200))
            value = st.slider("Giá trị thay thế (Value)", 0, 255, 255)
            preserve_bg = st.checkbox("Bảo toàn nền (Preserve Background)", value=True)
            result_img = gray_level_slicing(original_img, low=low, high=high, value=value, preserve_background=preserve_bg)
        elif algo == "Trích xuất mặt phẳng bit (Bit-plane Slicing)":
            st.write("🔧 **Trích xuất và khôi phục mặt phẳng bit**")
            bit_mode = st.radio("Chế độ hiển thị mặt phẳng bit", [
                "Hiển thị một mặt phẳng bit duy nhất",
                "Tái tạo ảnh từ bit N đến 7"
            ])
            bit_idx = st.slider("Mặt phẳng bit (0: LSB, 7: MSB)", 0, 7, 7)
            if bit_mode == "Hiển thị một mặt phẳng bit duy nhất":
                result_img = bit_plane_slicing(original_img, bit_idx=bit_idx)
            else:
                result_img = reconstruct_bit_planes(original_img, start_bit=bit_idx)
        elif algo == "Cân bằng Lược đồ xám (Histogram Equalization)":
            result_img = histogram_equalization(original_img)
        elif algo == "Bộ lọc Gaussian làm mượt (Gaussian Blur)":
            st.write("🔧 **Cấu hình bộ lọc Gaussian**")
            k_size = st.slider("Kích thước ma trận lọc (Kernel Size - Phải là số lẻ)", 3, 25, 5, 2)
            sigma = st.slider("Độ lệch chuẩn (Sigma - 0 nghĩa là tự động tính)", 0.0, 10.0, 0.0, 0.1)
            result_img = gaussian_smoothing_filter(original_img, kernel_size=k_size, sigma=sigma)
        elif algo == "Bộ lọc Box filter tùy biến":
            st.write("🔧 **Cấu hình bộ lọc Box**")
            width = st.slider("Chiều rộng ma trận (Width)", 1, 25, 5)
            height = st.slider("Chiều cao ma trận (Height)", 1, 25, 5)
            normalize = st.checkbox("Chuẩn hóa ma trận (Normalize)", value=True)
            result_img = box_filter(original_img, width=width, height=height, normalize=normalize)
        elif algo == "Làm sắc nét ảnh biên (Bộ lọc Laplacian)":
            result_img = sharpening_laplacian(original_img)
        elif algo == "Làm sắc nét đạo hàm bậc nhất (Sobel, Prewitt, Roberts)":
            st.write("🔧 **Cấu hình đạo hàm bậc nhất**")
            operator_type = st.selectbox("Chọn toán tử đạo hàm", ["Sobel", "Prewitt", "Roberts"])
            op_mode = st.selectbox("Chọn chế độ đầu ra", ["Chỉ hiển thị biên (Edge magnitude)", "Làm sắc nét (Sharpened Image)"])
            weight = 0.5
            if op_mode == "Làm sắc nét (Sharpened Image)":
                weight = st.slider("Hệ số cộng biên (Weight)", 0.1, 5.0, 0.5, 0.1)
            result_img = first_derivative_sharpening(original_img, operator_type=operator_type, mode=op_mode, weight=weight)

        # Hiển thị trực quan 2 ảnh song song để so sánh
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Hình Ảnh Gốc Ban Đầu", use_column_width=True)
        with col2:
            st.image(result_img, caption=f"Kết Quả Tăng Cường: {algo}", use_column_width=True)

    elif mode == "Phục hồi ảnh (Chapter 5)":
        st.header("🛠️ Phân Hệ: Mô Phỏng Suy Hao & Khôi Phục Ảnh")
        
        # Bước 1: Giả lập môi trường truyền tin gây nhiễu ảnh gốc
        st.subheader("Bước 1: Cấu hình gây lỗi nhiễu (Degradation Model)")
        noise_type = st.radio("Chọn dạng nhiễu môi trường tác động", [
            "Ảnh sạch (Không nhiễu)", 
            "Nhiễu Gaussian", 
            "Nhiễu Muối Tiêu (Salt & Pepper)",
            "Nhiễu Đồng Đều (Uniform)",
            "Nhiễu Erlang (Gamma)",
            "Nhiễu Rayleigh",
            "Nhiễu Tuần Hoàn (Periodic)"
        ])
        
        noisy_img = original_img.copy()
        if noise_type == "Nhiễu Gaussian":
            sigma = st.slider("Độ lệch chuẩn mức độ nhiễu (Sigma)", 5, 80, 25, 5)
            noisy_img = add_gaussian_noise(original_img, sigma=sigma)
        elif noise_type == "Nhiễu Muối Tiêu (Salt & Pepper)":
            prob = st.slider("Mật độ nhiễu hạt xung tác động (Xác suất)", 0.01, 0.15, 0.03, 0.01)
            noisy_img = add_salt_and_pepper_noise(original_img, salt_prob=prob, pepper_prob=prob)
        elif noise_type == "Nhiễu Đồng Đều (Uniform)":
            st.info("💡 Điểm ảnh sẽ cộng thêm giá trị ngẫu nhiên trong đoạn [a, b].")
            a_val = st.slider("Cực tiểu a", -100, 0, -20, 5)
            b_val = st.slider("Cực đại b", 0, 100, 20, 5)
            noisy_img = add_uniform_noise(original_img, a=a_val, b=b_val)
        elif noise_type == "Nhiễu Erlang (Gamma)":
            st.info("💡 Nhiễu Erlang/Gamma có phân bố lệch phải.")
            scale_val = st.slider("Hệ số tỉ lệ (Scale / a)", 1.0, 50.0, 10.0, 1.0)
            shape_val = st.slider("Số nguyên hình thái (Shape / b - bậc)", 1, 10, 2, 1)
            noisy_img = add_erlang_noise(original_img, a=scale_val, b=shape_val)
        elif noise_type == "Nhiễu Rayleigh":
            st.info("💡 Nhiễu Rayleigh phân bố không đối xứng, đặc trưng cho nhiễu radar.")
            scale_val = st.slider("Hệ số Rayleigh (Scale)", 5.0, 50.0, 15.0, 1.0)
            offset_val = st.slider("Độ dịch vị trí (Offset / a)", 0.0, 100.0, 0.0, 5.0)
            noisy_img = add_rayleigh_noise(original_img, a=offset_val, scale=scale_val)
        elif noise_type == "Nhiễu Tuần Hoàn (Periodic)":
            st.info("💡 Nhiễu sóng sin gây ra các sọc sọc trên ảnh và tạo ra các đốm nhiễu đối xứng trong phổ FFT.")
            amplitude = st.slider("Biên độ sóng sin (Amplitude)", 5.0, 100.0, 20.0, 5.0)
            u0 = st.slider("Tần số đứng (u0)", 1.0, 50.0, 10.0, 1.0)
            v0 = st.slider("Tần số ngang (v0)", 1.0, 50.0, 10.0, 1.0)
            noisy_img = add_periodic_noise(original_img, amplitude=amplitude, u0=u0, v0=v0)
            
        # Bước 2: Chọn giải thuật bộ lọc để tối ưu khôi phục lại cấu trúc ảnh ban đầu
        st.subheader("Bước 2: Chọn bộ lọc không gian khôi phục (Restoration Filter)")
        filter_type = st.selectbox("Chọn giải thuật lọc", [
            "Arithmetic Mean Filter (Trung bình số học)",
            "Geometric Mean Filter (Trung bình hình học)",
            "Harmonic Mean Filter (Trung bình điều hòa)",
            "Contraharmonic Mean Filter (Trung bình nghịch đảo điều hòa)",
            "Median Filter (Bộ lọc trung vị thống kê)",
            "Max Filter (Bộ lọc cực đại)",
            "Min Filter (Bộ lọc cực tiểu)",
            "Midpoint Filter (Bộ lọc điểm giữa)",
            "Alpha-trimmed Mean Filter (Bộ lọc trung vị điều chỉnh)"
        ])
        k_size = st.slider("Kích thước ma trận lọc quét qua ảnh (Kernel Size - Số lẻ)", 3, 11, 3, 2)
        
        # Cấu hình tham số riêng biệt cho bộ lọc nâng cao
        Q = 1.5
        d = 2
        if filter_type == "Contraharmonic Mean Filter (Trung bình nghịch đảo điều hòa)":
            st.info("💡 Q > 0: khử nhiễu Hạt tiêu (Pepper). Q < 0: khử nhiễu Hạt muối (Salt).")
            Q = st.slider("Hệ số lọc Q", -5.0, 5.0, 1.5, 0.1)
        elif filter_type == "Alpha-trimmed Mean Filter (Bộ lọc trung vị điều chỉnh)":
            max_d = (k_size * k_size) - 1
            st.info(f"💡 Số lượng phần tử cắt xén d phải chẵn, nằm trong khoảng [0, {max_d}].")
            d = st.slider("Số điểm ảnh cắt bỏ d", 0, max_d, min(2, max_d), 2)
        
        # Xử lý lọc suy giảm lỗi nhiễu
        restored_img = noisy_img.copy()
        if filter_type == "Arithmetic Mean Filter (Trung bình số học)":
            restored_img = arithmetic_mean_filter(noisy_img, kernel_size=k_size)
        elif filter_type == "Geometric Mean Filter (Trung bình hình học)":
            restored_img = geometric_mean_filter(noisy_img, kernel_size=k_size)
        elif filter_type == "Median Filter (Bộ lọc trung vị thống kê)":
            restored_img = median_filter(noisy_img, kernel_size=k_size)
        elif filter_type == "Max Filter (Bộ lọc cực đại)":
            restored_img = max_filter(noisy_img, kernel_size=k_size)
        elif filter_type == "Min Filter (Bộ lọc cực tiểu)":
            restored_img = min_filter(noisy_img, kernel_size=k_size)

        # Đo lường chính xác các chỉ số kỹ thuật phục vụ đánh giá so sánh hiệu năng
        mse_noisy = calculate_mse(original_img, noisy_img)
        snr_noisy = calculate_snr(original_img, noisy_img)
        
        mse_restored = calculate_mse(original_img, restored_img)
        snr_restored = calculate_snr(original_img, restored_img)

        # Hiển thị biểu đồ 3 cột song song so sánh tiến trình khôi phục
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_img, caption="Ảnh Gốc Gốc ban đầu", use_column_width=True)
            st.metric("MSE Gốc", "0.00")
        with col2:
            st.image(noisy_img, caption=f"Ảnh Nhiễu Suy Hao ({noise_type})", use_column_width=True)
            st.metric("MSE Sau Nhiễu", f"{mse_noisy:.2f}", delta=f"{mse_noisy:.2f}", delta_color="inverse")
            st.write(f"📊 **SNR Nhiễu:** {snr_noisy:.4f}")
        with col3:
            st.image(restored_img, caption=f"Ảnh Đã Phục Hồi ({filter_type})", use_column_width=True)
            # Nếu chỉ số MSE phục hồi nhỏ hơn MSE nhiễu, delta hiển thị trạng thái cải thiện tốt
            st.metric("MSE Sau Phục Hồi", f"{mse_restored:.2f}", delta=f"{restored_img.astype(float).mean() - noisy_img.astype(float).mean():.2f} (Độ lệch tối ưu)", delta_color="normal")
            st.write(f"📈 **SNR Đích Khôi Phục:** {snr_restored:.4f}")
            
            if mse_restored < mse_noisy:
                st.success("Cải thiện thành công! Chỉ số lỗi bình phương (MSE) đã giảm đáng kể.")
                
    elif mode == "Xử lý miền tần số (Chapter 4 & 5)":
        st.header("🌐 Phân Hệ: Xử Lý & Phục Hồi Trong Miền Tần Số (Frequency Domain)")
        
        # Convert sang grayscale
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        orig_spectrum, fshift = get_fft_spectrum(original_gray)
        
        algo_freq = st.selectbox("Chọn giải thuật miền tần số", [
            "Biến đổi Fourier & Hiển thị phổ (FFT Spectrum)",
            "Bộ lọc Chặn dải / Thông dải (Bandreject / Bandpass)",
            "Bộ lọc Notch (Notch Reject / Pass)",
            "Phục hồi bằng bộ lọc Wiener (Wiener Filter)"
        ])
        
        if algo_freq == "Biến đổi Fourier & Hiển thị phổ (FFT Spectrum)":
            st.write("Giải thuật hiển thị phổ tần số của ảnh sử dụng biến đổi Fourier nhanh (FFT). Vùng sáng ở tâm đại diện cho tần số thấp, các vùng xa tâm đại diện cho tần số cao.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_gray, caption="Ảnh gốc xám (Grayscale)", use_column_width=True)
            with col2:
                st.image(orig_spectrum, caption="Phổ biên độ tần số (Magnitude Spectrum)", use_column_width=True)
                
        elif algo_freq == "Bộ lọc Chặn dải / Thông dải (Bandreject / Bandpass)":
            st.subheader("Cấu hình bộ lọc Chặn dải / Thông dải")
            filter_class = st.selectbox("Dạng bộ lọc", ["Ideal", "Butterworth", "Gaussian"])
            pass_type = st.selectbox("Kiểu lọc", ["Chặn dải (Bandreject)", "Thông dải (Bandpass)"])
            
            D0 = st.slider("Tần số cắt trung tâm (D0)", 1, min(original_gray.shape) // 2, 50)
            W = st.slider("Bề rộng băng thông (W)", 1, 100, 20)
            
            n = 1
            if filter_class == "Butterworth":
                n = st.slider("Bậc của bộ lọc Butterworth (n)", 1, 5, 1)
                
            # Tính toán mặt nạ bộ lọc
            mask = bandreject_filter_mask(original_gray.shape, D0=D0, W=W, filter_type=filter_class, n=n)
            if pass_type == "Thông dải (Bandpass)":
                mask = 1.0 - mask
                
            # Áp dụng bộ lọc lên FFT
            gshift = fshift * mask
            result_gray = reconstruct_from_fft(gshift)
            
            # Chuẩn bị hiển thị phổ
            filter_mask_display = (mask * 255).astype(np.uint8)
            filtered_spectrum = 20 * np.log(np.abs(gshift) + 1)
            mag_max = np.max(filtered_spectrum)
            if mag_max > 0:
                filtered_spectrum = (filtered_spectrum / mag_max) * 255
            filtered_spectrum_display = filtered_spectrum.astype(np.uint8)
            
            # Hiển thị kết quả 3 cột
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(original_gray, caption="Ảnh gốc xám", use_column_width=True)
                st.image(orig_spectrum, caption="Phổ tần số gốc", use_column_width=True)
            with col2:
                st.image(filter_mask_display, caption=f"Bộ lọc {filter_class} {pass_type} H(u,v)", use_column_width=True)
                st.image(filtered_spectrum_display, caption="Phổ tần số sau khi lọc", use_column_width=True)
            with col3:
                st.image(result_gray, caption="Ảnh kết quả (Inverse FFT)", use_column_width=True)
                
        elif algo_freq == "Bộ lọc Notch (Notch Reject / Pass)":
            st.subheader("Cấu hình bộ lọc Notch")
            st.write("Bộ lọc Notch chặn đứng (Reject) hoặc chỉ giữ lại (Pass) các thành phần tần số xác định (giúp khử nhiễu tuần hoàn).")
            
            filter_class = st.selectbox("Dạng bộ lọc", ["Ideal", "Butterworth", "Gaussian"])
            notch_mode = st.selectbox("Chế độ Notch", ["Reject (Chặn tần số nhiễu)", "Pass (Chỉ giữ lại tần số nhiễu)"])
            
            D0 = st.slider("Bán kính vùng Notch (D0)", 1, 50, 10)
            uk = st.slider("Lệch tọa độ trục U (uk)", -original_gray.shape[0]//2, original_gray.shape[0]//2, 30)
            vk = st.slider("Lệch tọa độ trục V (vk)", -original_gray.shape[1]//2, original_gray.shape[1]//2, 30)
            
            n = 1
            if filter_class == "Butterworth":
                n = st.slider("Bậc của bộ lọc (n)", 1, 5, 1)
                
            mode_str = "Reject" if notch_mode == "Reject (Chặn tần số nhiễu)" else "Pass"
            mask = notch_filter_mask(original_gray.shape, D0=D0, uk=uk, vk=vk, filter_type=filter_class, n=n, mode=mode_str)
            
            gshift = fshift * mask
            result_gray = reconstruct_from_fft(gshift)
            
            filter_mask_display = (mask * 255).astype(np.uint8)
            filtered_spectrum = 20 * np.log(np.abs(gshift) + 1)
            mag_max = np.max(filtered_spectrum)
            if mag_max > 0:
                filtered_spectrum = (filtered_spectrum / mag_max) * 255
            filtered_spectrum_display = filtered_spectrum.astype(np.uint8)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(original_gray, caption="Ảnh gốc xám", use_column_width=True)
                st.image(orig_spectrum, caption="Phổ tần số gốc", use_column_width=True)
            with col2:
                st.image(filter_mask_display, caption=f"Bộ lọc Notch {filter_class} ({notch_mode})", use_column_width=True)
                st.image(filtered_spectrum_display, caption="Phổ tần số sau lọc", use_column_width=True)
            with col3:
                st.image(result_gray, caption="Ảnh kết quả (Inverse FFT)", use_column_width=True)
                
        elif algo_freq == "Phục hồi bằng bộ lọc Wiener (Wiener Filter)":
            st.subheader("Phục hồi ảnh bị nhòe và nhiễu bằng bộ lọc Wiener")
            st.write("Quy trình giả lập: Ảnh gốc -> Làm nhòe bằng Gaussian Lowpass (Blur) -> Thêm nhiễu Gaussian -> Khôi phục bằng bộ lọc Wiener.")
            
            D0_blur = st.slider("Mức độ làm nhòe (D0) - Càng nhỏ ảnh càng nhòe", 5, 100, 30)
            noise_sigma = st.slider("Mức độ nhiễu bổ sung (Sigma)", 0, 50, 10)
            K = st.slider("SNR nghịch đảo (Tham số K của Wiener Filter)", 0.0001, 0.5000, 0.0100, 0.0005, format="%.4f")
            
            # Thực hiện giả lập và khôi phục
            noisy_blurred, restored, blur_filter_mask, restored_spec = wiener_filter_restoration(
                original_gray, D0=D0_blur, K=K, noise_sigma=noise_sigma
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(original_gray, caption="Ảnh gốc xám", use_column_width=True)
                st.image(orig_spectrum, caption="Phổ tần số gốc", use_column_width=True)
            with col2:
                st.image(noisy_blurred, caption="Ảnh nhòe + Nhiễu", use_column_width=True)
                # Tính phổ của ảnh nhòe nhiễu để trực quan
                fshift_noisy_mag, _ = get_fft_spectrum(noisy_blurred)
                st.image(fshift_noisy_mag, caption="Phổ ảnh nhòe nhiễu", use_column_width=True)
            with col3:
                st.image(blur_filter_mask, caption="Mặt nạ làm nhòe H(u,v)", use_column_width=True)
                st.image(restored_spec, caption="Phổ ảnh phục hồi", use_column_width=True)
            with col4:
                st.image(restored, caption="Ảnh khôi phục (Wiener)", use_column_width=True)
                # Tính toán MSE/SNR để người dùng tự đánh giá
                from utils import calculate_mse, calculate_snr
                mse_nb = calculate_mse(original_gray, noisy_blurred)
                snr_nb = calculate_snr(original_gray, noisy_blurred)
                mse_rest = calculate_mse(original_gray, restored)
                snr_rest = calculate_snr(original_gray, restored)
                
                st.metric("MSE Phục hồi", f"{mse_rest:.2f}", delta=f"{mse_rest - mse_nb:.2f}", delta_color="inverse")
                st.write(f"📊 **SNR Nhòe+Nhiễu:** {snr_nb:.4f}")
                st.write(f"📈 **SNR Sau Wiener:** {snr_rest:.4f}")
                if mse_rest < mse_nb:
                    st.success("Tối ưu Wiener thành công!")
else:
    st.info("💡 Vui lòng thực hiện kéo thả hoặc tải một hình ảnh từ thanh menu bên trái (Sidebar) để trải nghiệm ứng dụng.")