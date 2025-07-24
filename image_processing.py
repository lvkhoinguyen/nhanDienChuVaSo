# file: image_processing.py

import numpy as np
import cv2

# --- CÁC KERNEL TÍCH CHẬP ---

# 1. Kernel làm sắc nét (Sharpen)
KERNEL_SHARPEN = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])

# 2. Kernel phát hiện biên (Sobel)
# Hướng X: phát hiện các cạnh dọc
KERNEL_SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
# Hướng Y: phát hiện các cạnh ngang
KERNEL_SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# 3. Kernel làm mờ Gauss (Gaussian Blur) - Giảm nhiễu
# Đây là kernel 5x5 đã được chuẩn hóa (tổng các phần tử là 1)
KERNEL_GAUSSIAN = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / 256.0


# --- CÁC HÀM XỬ LÝ ẢNH BẰNG TÍCH CHẬP ---

def apply_convolution(image, kernel):
    """
    Áp dụng một phép tích chập lên ảnh sử dụng kernel cho trước.

    Args:
        image (numpy.array): Ảnh đầu vào (thường là ảnh thang xám).
        kernel (numpy.array): Ma trận kernel để thực hiện tích chập.

    Returns:
        numpy.array: Ảnh sau khi đã áp dụng tích chập.
    """
    # cv2.filter2D là hàm được tối ưu hóa cao để thực hiện phép tích chập 2D.
    # -1 trong tham số thứ hai (ddepth) có nghĩa là ảnh đầu ra sẽ có cùng độ sâu (data type) với ảnh đầu vào.
    return cv2.filter2D(image, -1, kernel)


def sharpen_image(image):
    """Làm sắc nét ảnh bằng cách nhấn mạnh sự khác biệt giữa các pixel."""
    return apply_convolution(image, KERNEL_SHARPEN)


def edge_detection(image):
    """
    Phát hiện các đường biên trong ảnh bằng toán tử Sobel.
    Toán tử này tính toán đạo hàm của ảnh theo hai hướng X và Y.
    """
    # Tính toán đạo hàm theo hướng X (cạnh dọc)
    edges_x = apply_convolution(image, KERNEL_SOBEL_X)

    # Tính toán đạo hàm theo hướng Y (cạnh ngang)
    edges_y = apply_convolution(image, KERNEL_SOBEL_Y)

    # Kết hợp kết quả từ hai hướng để có được bản đồ biên cạnh hoàn chỉnh.
    # cv2.addWeighted cộng hai ảnh với trọng số nhất định. Ở đây ta lấy 50% từ mỗi hướng.
    return cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)


def noise_reduction(image):
    """Giảm nhiễu trong ảnh bằng cách làm mờ theo phân phối Gaussian."""
    return apply_convolution(image, KERNEL_GAUSSIAN)


def enhance_text(image):
    """
    Một quy trình hoàn chỉnh để tăng cường chất lượng văn bản trong ảnh.
    Kết hợp nhiều kỹ thuật tích chập để cho ra kết quả tốt nhất.
    """
    # Bước 1: Giảm nhiễu hạt và các chi tiết không mong muốn bằng Gaussian Blur.
    denoised = noise_reduction(image)

    # Bước 2: Làm sắc nét ảnh đã giảm nhiễu để các ký tự trở nên rõ ràng hơn.
    sharpened = sharpen_image(denoised)

    # Tùy chọn (có thể bỏ qua): Bạn có thể phát hiện biên và kết hợp để làm nổi bật hơn nữa
    # edges = edge_detection(sharpened)
    # enhanced = cv2.addWeighted(sharpened, 0.8, edges, 0.2, 0)
    # return enhanced

    # Trả về ảnh đã được làm sắc nét, đây là bước quan trọng nhất
    return sharpened