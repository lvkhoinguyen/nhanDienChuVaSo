# file: train.py

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình đẹp mắt

# Import các hàm xử lý ảnh từ file image_processing.py
# Đảm bảo file image_processing.py nằm cùng thư mục với train.py
try:
    from image_processing import enhance_text
except ImportError:
    print("Lỗi: Không tìm thấy file 'image_processing.py'.")
    print("Vui lòng đảm bảo file này tồn tại trong cùng thư mục.")
    exit()  # Thoát chương trình nếu không tìm thấy file cần thiết


# --- 1. ĐỊNH NGHĨA CÁC LỚP KÝ TỰ ---
# Tạo ra danh sách 62 lớp: 0-9, A-Z, a-z
# Thứ tự này phải khớp với thứ tự các thư mục 'Samplexxx'
def get_classes():
    """Tạo và trả về một danh sách chứa tất cả các lớp ký tự."""
    classes = []
    # Các chữ số từ 0 đến 9
    for i in range(10):
        classes.append(str(i))
    # Các chữ cái viết hoa A-Z (mã ASCII từ 65 đến 90)
    for i in range(26):
        classes.append(chr(65 + i))
    # Các chữ cái viết thường a-z (mã ASCII từ 97 đến 122)
    for i in range(26):
        classes.append(chr(97 + i))
    return classes


# --- 2. HÀM TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---
def load_data(data_path="data/English/Fnt", img_size=(28, 28), max_images_per_class=1000):
    """
    Tải dữ liệu ảnh từ cấu trúc thư mục, áp dụng tiền xử lý và trả về dưới dạng mảng numpy.

    Args:
        data_path (str): Đường dẫn đến thư mục gốc chứa các thư mục Sample.
        img_size (tuple): Kích thước (width, height) để resize ảnh.
        max_images_per_class (int): Số lượng ảnh tối đa lấy từ mỗi lớp để tránh mất cân bằng và quá tải bộ nhớ.

    Returns:
        tuple: (X, y) nếu thành công, (None, None) nếu thất bại.
               X là mảng numpy chứa dữ liệu ảnh, y là mảng numpy chứa nhãn.
    """
    print(f"Bắt đầu tải dữ liệu từ: {data_path}")

    if not os.path.exists(data_path):
        print(f"LỖI: Không tìm thấy thư mục dữ liệu tại đường dẫn: {data_path}")
        return None, None

    classes = get_classes()
    X, y = [], []

    # Lấy danh sách thư mục con và sắp xếp để đảm bảo đúng thứ tự
    try:
        sample_dirs = sorted(
            [d for d in os.listdir(data_path) if d.startswith('Sample') and os.path.isdir(os.path.join(data_path, d))],
            key=lambda x: int(x[6:])  # Sắp xếp theo số thứ tự của Samplexxx
        )
    except ValueError:
        print(f"LỖI: Tên thư mục trong {data_path} không đúng định dạng 'Samplexxx'.")
        return None, None

    # Sử dụng tqdm để hiển thị thanh tiến trình
    for i, sample_dir in enumerate(tqdm(sample_dirs, desc="Đang xử lý các lớp")):
        if i >= len(classes):
            print(f"Cảnh báo: Có nhiều thư mục Sample hơn số lớp định nghĩa. Bỏ qua {sample_dir}.")
            break

        label = classes[i]
        sample_path = os.path.join(data_path, sample_dir)

        image_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for img_file in image_files[:max_images_per_class]:
            img_path = os.path.join(sample_path, img_file)
            try:
                # 1. Đọc ảnh dưới dạng thang xám
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    # 2. **TIỀN XỬ LÝ QUAN TRỌNG**: Tăng cường chất lượng ảnh
                    enhanced_img = enhance_text(img)

                    # 3. Thay đổi kích thước ảnh về kích thước chuẩn
                    resized_img = cv2.resize(enhanced_img, img_size)

                    # 4. Chuẩn hóa giá trị pixel về khoảng [0.0, 1.0]
                    normalized_img = resized_img.astype('float32') / 255.0

                    X.append(normalized_img)
                    y.append(label)

            except Exception as e:
                print(f"\nCảnh báo: Lỗi khi đọc hoặc xử lý ảnh {img_path}: {e}")
                continue

    if not X:
        print("LỖI: Không có dữ liệu ảnh hợp lệ nào được tải. Vui lòng kiểm tra lại cấu trúc thư mục và file ảnh.")
        return None, None

    print("\nChuyển đổi dữ liệu sang định dạng NumPy array...")
    X = np.array(X)
    y = np.array(y)

    # Thêm một chiều cho kênh màu (ví dụ: (N, 28, 28) -> (N, 28, 28, 1))
    # Đây là định dạng đầu vào mà các lớp Conv2D của Keras yêu cầu
    X = X.reshape(X.shape[0], img_size[0], img_size[1], 1)

    print(f"Tải dữ liệu hoàn tất! Tìm thấy {len(X)} ảnh.")
    print(f"Kích thước tập dữ liệu ảnh (X): {X.shape}")
    print(f"Kích thước tập nhãn (y): {y.shape}")

    return X, y


# --- 3. HÀM TẠO KIẾN TRÚC MÔ HÌNH CNN ---
def create_cnn_model(num_classes, input_shape=(28, 28, 1)):
    """
    Xây dựng kiến trúc mô hình Mạng nơ-ron Tích chập (CNN).
    """
    print("Đang tạo kiến trúc mô hình CNN...")

    model = keras.Sequential([
        # Lớp đầu vào, xác định hình dạng của một ảnh
        keras.layers.Input(shape=input_shape),

        # Khối Tích chập 1
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Khối Tích chập 2
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Lớp Flatten để chuyển từ dạng 2D sang vector 1D
        keras.layers.Flatten(),

        # Lớp Dropout để chống overfitting (học vẹt)
        keras.layers.Dropout(0.5),

        # Lớp kết nối đầy đủ (Fully Connected)
        keras.layers.Dense(128, activation='relu'),

        # Lớp đầu ra với hàm kích hoạt softmax cho bài toán phân loại đa lớp
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Biên dịch mô hình: xác định trình tối ưu, hàm mất mát và các thước đo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()  # In tóm tắt kiến trúc mô hình
    return model


# --- 4. HÀM HUẤN LUYỆN VÀ LƯU MÔ HÌNH ---
def train_and_save_model():
    """Hàm chính điều phối toàn bộ quy trình."""

    # Tải và xử lý dữ liệu
    X_data, y_labels = load_data()
    if X_data is None:
        return

    # Mã hóa nhãn: chuyển nhãn từ dạng chữ ('A', 'b', '5') sang số nguyên (0, 1, 2...)
    print("Đang mã hóa nhãn...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    num_classes = len(np.unique(y_encoded))
    print(f"Số lượng lớp (ký tự) duy nhất: {num_classes}")

    # Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm thử (20%)
    print("Đang chia dữ liệu thành tập train và test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_encoded,
        test_size=0.2,  # Tỷ lệ của tập kiểm thử
        random_state=42,  # Đảm bảo kết quả chia luôn giống nhau mỗi lần chạy
        stratify=y_encoded  # Giữ tỷ lệ phân phối các lớp trong cả hai tập
    )

    print(f"Kích thước tập huấn luyện: {X_train.shape}, {y_train.shape}")
    print(f"Kích thước tập kiểm thử: {X_test.shape}, {y_test.shape}")

    # Tạo mô hình
    model = create_cnn_model(num_classes)

    # Định nghĩa các callbacks
    # EarlyStopping: Dừng huấn luyện sớm nếu hiệu suất trên tập validation không cải thiện
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # ModelCheckpoint: Lưu lại mô hình tốt nhất trong quá trình huấn luyện
    os.makedirs("models", exist_ok=True)  # Tạo thư mục 'models' nếu chưa có
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "models/best_handwriting_model.h5",
        save_best_only=True,
        monitor='val_accuracy'
    )

    # Bắt đầu quá trình huấn luyện
    print("\n--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ---")
    history = model.fit(
        X_train, y_train,
        batch_size=64,  # Xử lý 64 ảnh trong mỗi lần cập nhật trọng số
        epochs=20,  # Lặp qua toàn bộ tập dữ liệu 20 lần
        validation_data=(X_test, y_test),  # Dữ liệu để đánh giá sau mỗi epoch
        callbacks=[early_stopping, model_checkpoint],  # Sử dụng các callbacks đã định nghĩa
        verbose=1  # Hiển thị thông tin huấn luyện chi tiết
    )

    print("\n--- HUẤN LUYỆN HOÀN TẤT ---")

    # Đánh giá hiệu suất cuối cùng của mô hình trên tập kiểm thử
    print("Đánh giá mô hình trên tập dữ liệu test...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"==> Độ chính xác trên tập kiểm thử: {test_acc:.4f}")
    print(f"==> Mất mát trên tập kiểm thử: {test_loss:.4f}")

    # Lưu mô hình cuối cùng và bộ mã hóa nhãn
    print("Đang lưu mô hình cuối cùng và bộ mã hóa nhãn...")
    model.save("models/final_handwriting_model.h5")
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nĐã lưu thành công mô hình và bộ mã hóa vào thư mục 'models/'")
    print("Các file đã được tạo:")
    print("- models/best_handwriting_model.h5 (Mô hình có độ chính xác validation tốt nhất)")
    print("- models/final_handwriting_model.h5 (Mô hình ở epoch cuối cùng)")
    print("- models/label_encoder.pkl (Bộ chuyển đổi nhãn chữ sang số)")


# --- 5. ĐIỂM KHỞI CHẠY CHÍNH CỦA CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    train_and_save_model()