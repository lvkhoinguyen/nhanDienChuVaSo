import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import threading
import time
from collections import Counter
from sklearn.preprocessing import LabelEncoder


class HandwritingTextRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện Văn bản Viết tay - AI Text Scanner")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')

        # Biến lưu trữ
        self.model = None
        self.label_encoder = None
        self.image_path = None
        self.original_image = None
        self.processed_image = None

        # Classes mapping (0-9, A-Z, a-z)
        self.classes = []
        for i in range(10):  # 0-9
            self.classes.append(str(i))
        for i in range(26):  # A-Z
            self.classes.append(chr(65 + i))
        for i in range(26):  # a-z
            self.classes.append(chr(97 + i))

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="📄 NHẬN DIỆN VĂN BẢN VIẾT TAY",
                               font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 15))

        # Control Panel
        control_frame = tk.LabelFrame(main_frame, text="Bảng điều khiển",
                                      font=('Arial', 12, 'bold'), bg='#f0f0f0')
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Model controls
        model_frame = tk.Frame(control_frame, bg='#f0f0f0')
        model_frame.pack(pady=10)

        self.train_btn = tk.Button(model_frame, text="🎯 Huấn luyện Model",
                                   command=self.start_training, bg='#3498db', fg='white',
                                   font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.train_btn.pack(side=tk.LEFT, padx=3)

        self.load_model_btn = tk.Button(model_frame, text="📂 Tải Model",
                                        command=self.load_model, bg='#9b59b6', fg='white',
                                        font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.load_model_btn.pack(side=tk.LEFT, padx=3)

        self.save_model_btn = tk.Button(model_frame, text="💾 Lưu Model",
                                        command=self.save_model, bg='#27ae60', fg='white',
                                        font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.save_model_btn.pack(side=tk.LEFT, padx=3)

        # Image processing controls
        process_frame = tk.Frame(control_frame, bg='#f0f0f0')
        process_frame.pack(pady=10)

        self.upload_btn = tk.Button(process_frame, text="🖼️ Chọn ảnh văn bản",
                                    command=self.upload_image, bg='#e74c3c', fg='white',
                                    font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.upload_btn.pack(side=tk.LEFT, padx=3)

        self.process_btn = tk.Button(process_frame, text="⚙️ Xử lý ảnh",
                                     command=self.process_image, bg='#f39c12', fg='white',
                                     font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.process_btn.pack(side=tk.LEFT, padx=3)

        self.scan_btn = tk.Button(process_frame, text="🔍 Scan văn bản",
                                  command=self.scan_text, bg='#e67e22', fg='white',
                                  font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.scan_btn.pack(side=tk.LEFT, padx=3)

        self.clear_btn = tk.Button(process_frame, text="🗑️ Xóa tất cả",
                                   command=self.clear_all, bg='#95a5a6', fg='white',
                                   font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=3)

        # Settings frame
        settings_frame = tk.Frame(control_frame, bg='#f0f0f0')
        settings_frame.pack(pady=10)

        tk.Label(settings_frame, text="Ngưỡng nhị phân:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=127)
        self.threshold_scale = tk.Scale(settings_frame, from_=50, to=200, orient='horizontal',
                                        variable=self.threshold_var, length=150, bg='#f0f0f0')
        self.threshold_scale.pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Kích thước tối thiểu:", font=('Arial', 10), bg='#f0f0f0').pack(side=tk.LEFT,
                                                                                                      padx=(20, 0))
        self.min_area_var = tk.IntVar(value=100)
        self.min_area_scale = tk.Scale(settings_frame, from_=50, to=500, orient='horizontal',
                                       variable=self.min_area_var, length=150, bg='#f0f0f0')
        self.min_area_scale.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        # Status label
        self.status_label = tk.Label(control_frame, text="Sẵn sàng",
                                     font=('Arial', 10), bg='#f0f0f0', fg='#2c3e50')
        self.status_label.pack(pady=5)

        # Main content area
        content_frame = tk.Frame(main_frame, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Image display
        left_frame = tk.LabelFrame(content_frame, text="Ảnh và xử lý",
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Original image
        orig_frame = tk.Frame(left_frame, bg='#f0f0f0')
        orig_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(orig_frame, text="Ảnh gốc:", font=('Arial', 10, 'bold'), bg='#f0f0f0').pack(anchor='w', padx=5)
        self.original_label = tk.Label(orig_frame, text="Chưa có ảnh",
                                       bg='white', relief='sunken', bd=2)
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Processed image
        proc_frame = tk.Frame(left_frame, bg='#f0f0f0')
        proc_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(proc_frame, text="Ảnh đã xử lý:", font=('Arial', 10, 'bold'), bg='#f0f0f0').pack(anchor='w', padx=5)
        self.processed_label = tk.Label(proc_frame, text="Chưa xử lý",
                                        bg='white', relief='sunken', bd=2)
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Text results
        right_frame = tk.LabelFrame(content_frame, text="Kết quả nhận diện văn bản",
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Statistics frame
        stats_frame = tk.Frame(right_frame, bg='#f0f0f0')
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stats_label = tk.Label(stats_frame, text="Thống kê: Chưa có dữ liệu",
                                    font=('Arial', 10), bg='#f0f0f0', fg='#666666')
        self.stats_label.pack(anchor='w')

        # Text result display
        text_frame = tk.Frame(right_frame, bg='#f0f0f0')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tk.Label(text_frame, text="Văn bản được nhận diện:",
                 font=('Arial', 12, 'bold'), bg='#f0f0f0').pack(anchor='w', pady=(0, 5))

        # Text area with scrollbar
        self.result_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD,
                                                     font=('Arial', 11), height=15)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Copy and save buttons
        button_frame = tk.Frame(right_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.copy_btn = tk.Button(button_frame, text="📋 Copy văn bản",
                                  command=self.copy_text, bg='#17a2b8', fg='white',
                                  font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.copy_btn.pack(side=tk.LEFT, padx=3)

        self.save_text_btn = tk.Button(button_frame, text="💾 Lưu văn bản",
                                       command=self.save_text, bg='#28a745', fg='white',
                                       font=('Arial', 10, 'bold'), padx=15, pady=5)
        self.save_text_btn.pack(side=tk.LEFT, padx=3)

    def load_data(self):
        """Tải dữ liệu từ thư mục data/English/Fnt/"""
        X, y = [], []
        data_path = "data/English/Fnt"

        if not os.path.exists(data_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy thư mục dữ liệu: {data_path}")
            return None, None

        sample_dirs = sorted([d for d in os.listdir(data_path) if d.startswith('Sample')])

        for i, sample_dir in enumerate(sample_dirs):
            if i >= len(self.classes):
                break

            sample_path = os.path.join(data_path, sample_dir)
            if not os.path.isdir(sample_path):
                continue

            label = self.classes[i]
            image_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            # Giới hạn số lượng ảnh mỗi lớp để tránh quá tải
            max_images = 1000
            for img_file in image_files[:max_images]:
                img_path = os.path.join(sample_path, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (28, 28))
                        img = img.astype('float32') / 255.0
                        X.append(img)
                        y.append(label)
                except Exception as e:
                    print(f"Lỗi đọc ảnh {img_path}: {e}")
                    continue

        if len(X) == 0:
            messagebox.showerror("Lỗi", "Không tìm thấy dữ liệu ảnh hợp lệ!")
            return None, None

        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], 28, 28, 1)

        return X, y

    def create_model(self, num_classes):
        """Tạo model CNN"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def start_training(self):
        """Bắt đầu huấn luyện trong thread riêng"""
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()

    def train_model(self):
        """Huấn luyện model"""
        try:
            self.root.after(0, lambda: self.progress.start())
            self.root.after(0, lambda: self.status_label.config(text="Đang tải dữ liệu..."))

            X, y = self.load_data()
            if X is None:
                return

            self.root.after(0, lambda: self.status_label.config(text="Đang chuẩn bị dữ liệu..."))

            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            self.root.after(0, lambda: self.status_label.config(text="Đang huấn luyện model..."))

            num_classes = len(np.unique(y_encoded))
            self.model = self.create_model(num_classes)

            class TrainingCallback(keras.callbacks.Callback):
                def __init__(self, app):
                    self.app = app

                def on_epoch_end(self, epoch, logs=None):
                    status = f"Epoch {epoch + 1}/15 - Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f}"
                    self.app.root.after(0, lambda: self.app.status_label.config(text=status))

            history = self.model.fit(
                X_train, y_train,
                epochs=15,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[TrainingCallback(self)],
                verbose=0
            )

            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)

            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_label.config(
                text=f"Huấn luyện hoàn thành! Độ chính xác: {test_acc:.4f}"
            ))

            messagebox.showinfo("Thành công",
                                f"Huấn luyện hoàn thành!\nĐộ chính xác: {test_acc:.4f}")

        except Exception as e:
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_label.config(text="Lỗi huấn luyện"))
            messagebox.showerror("Lỗi", f"Lỗi huấn luyện: {str(e)}")

    def save_model(self):
        """Lưu model"""
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Chưa có model để lưu!")
            return

        try:
            os.makedirs("models", exist_ok=True)
            self.model.save("models/handwriting_text_model.h5")
            with open("models/text_label_encoder.pkl", "wb") as f:
                pickle.dump(self.label_encoder, f)

            messagebox.showinfo("Thành công", "Đã lưu model thành công!")
            self.status_label.config(text="Đã lưu model")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu model: {str(e)}")

    def load_model(self):
        """Tải model"""
        try:
            if os.path.exists("models/handwriting_text_model.h5") and os.path.exists("models/text_label_encoder.pkl"):
                self.model = keras.models.load_model("models/handwriting_text_model.h5")

                with open("models/text_label_encoder.pkl", "rb") as f:
                    self.label_encoder = pickle.load(f)

                messagebox.showinfo("Thành công", "Đã tải model thành công!")
                self.status_label.config(text="Đã tải model")
            else:
                messagebox.showwarning("Cảnh báo", "Không tìm thấy file model!")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải model: {str(e)}")

    def upload_image(self):
        """Upload ảnh văn bản"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh văn bản viết tay",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )

        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh. Vui lòng chọn ảnh khác.")
                return

            self.display_original_image()
            self.status_label.config(text="Đã tải ảnh")

    def display_original_image(self):
        """Hiển thị ảnh gốc"""
        if self.original_image is None:
            return

        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            # Resize để hiển thị
            height, width = rgb_image.shape[:2]
            max_size = 400

            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            resized = cv2.resize(rgb_image, (new_width, new_height))

            # Convert to PhotoImage
            image = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(image)

            self.original_label.config(image=photo, text="")
            self.original_label.image = photo

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {str(e)}")

    def process_image(self):
        """Xử lý ảnh để chuẩn bị cho việc phân đoạn"""
        if self.original_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh để xử lý!")
            return

        try:
            # Giảm kích thước ảnh nếu quá lớn để tăng tốc xử lý
            height, width = self.original_image.shape[:2]
            max_dimension = 2000
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(self.original_image, (int(width * scale), int(height * scale)))
            else:
                img = self.original_image.copy()

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

            self.processed_image = processed
            self.display_processed_image()
            self.status_label.config(text="Đã xử lý ảnh")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh: {str(e)}")

    def display_processed_image(self):
        """Hiển thị ảnh đã xử lý"""
        if self.processed_image is None:
            return

        try:
            # Resize để hiển thị
            height, width = self.processed_image.shape
            max_size = 400

            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            resized = cv2.resize(self.processed_image, (new_width, new_height))

            # Convert to PhotoImage
            image = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(image)

            self.processed_label.config(image=photo, text="")
            self.processed_label.image = photo

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh xử lý: {str(e)}")

    def find_characters(self):
        """Tìm và phân đoạn các ký tự từ ảnh"""
        if self.processed_image is None:
            return []

        # Find contours
        contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort contours
        char_contours = []
        min_area = self.min_area_var.get()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area and aspect ratio
            if area > min_area and w > 5 and h > 10 and h / w < 5 and w / h < 3:
                char_contours.append((x, y, w, h))

        # Sort by position (left to right, top to bottom)
        char_contours.sort(key=lambda x: (x[1] // 50, x[0]))  # Group by lines then sort by x

        return char_contours

    def extract_character(self, x, y, w, h):
        """Trích xuất và chuẩn bị ký tự cho nhận diện"""
        try:
            # Extract character region
            char_img = self.processed_image[y:y + h, x:x + w]

            # Add padding
            pad = 10
            padded = cv2.copyMakeBorder(char_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

            # Resize to 28x28
            resized = cv2.resize(padded, (28, 28))

            # Normalize
            normalized = resized.astype('float32') / 255.0

            # Reshape for model
            return normalized.reshape(1, 28, 28, 1)

        except Exception as e:
            print(f"Lỗi trích xuất ký tự: {e}")
            return None

    def scan_text(self):
        """Scan và nhận diện toàn bộ văn bản"""
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Chưa có model! Vui lòng huấn luyện hoặc tải model.")
            return

        if self.processed_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh đã xử lý! Vui lòng xử lý ảnh trước.")
            return

        try:
            self.progress.start()
            self.status_label.config(text="Đang phân đoạn và nhận diện...")

            # Find character contours
            char_contours = self.find_characters()

            if not char_contours:
                messagebox.showwarning("Cảnh báo", "Không tìm thấy ký tự nào! Thử điều chỉnh ngưỡng.")
                self.progress.stop()
                return

            # Recognize each character
            recognized_text = ""
            current_line_y = -1
            line_threshold = 30
            confidences = []

            for i, (x, y, w, h) in enumerate(char_contours):
                # Check if this is a new line
                if current_line_y == -1 or abs(y - current_line_y) > line_threshold:
                    if recognized_text and not recognized_text.endswith('\n'):
                        recognized_text += '\n'
                    current_line_y = y

                # Extract and predict character
                char_data = self.extract_character(x, y, w, h)
                if char_data is not None:
                    predictions = self.model.predict(char_data, verbose=0)
                    predicted_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_idx]

                    if confidence > 0.1:  # Minimum confidence threshold
                        predicted_char = self.label_encoder.inverse_transform([predicted_idx])[0]
                        recognized_text += predicted_char
                        confidences.append(confidence)
                    else:
                        recognized_text += '?'
                        confidences.append(0.1)

                # Update progress
                progress_percent = (i + 1) / len(char_contours) * 100
                self.status_label.config(text=f"Đang nhận diện... {progress_percent:.1f}%")
                self.root.update()

            # Display results
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', recognized_text)

            # Update statistics
            avg_confidence = np.mean(confidences) if confidences else 0
            char_count = len([c for c in recognized_text if c.strip()])
            word_count = len(recognized_text.split())
            line_count = recognized_text.count('\n') + 1

            stats_text = f"Ký tự: {char_count} | Từ: {word_count} | Dòng: {line_count} | Độ tin cậy TB: {avg_confidence:.2%}"
            self.stats_label.config(text=stats_text)

            self.progress.stop()
            self.status_label.config(text="Hoàn thành nhận diện văn bản")

        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Lỗi nhận diện")
            messagebox.showerror("Lỗi", f"Lỗi nhận diện văn bản: {str(e)}")

    def copy_text(self):
        """Copy văn bản đã nhận diện"""
        text = self.result_text.get('1.0', tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Thành công", "Đã copy văn bản vào clipboard!")
        else:
            messagebox.showwarning("Cảnh báo", "Không có văn bản để copy!")

    def save_text(self):
        """Lưu văn bản đã nhận diện ra file"""
        text = self.result_text.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("Cảnh báo", "Không có văn bản để lưu!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Lưu văn bản nhận diện",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            defaultextension=".txt"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Thành công", f"Đã lưu văn bản vào {file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file: {str(e)}")

    def clear_all(self):
        """Xóa tất cả dữ liệu"""
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.original_label.config(image=None, text="Chưa có ảnh")
        self.processed_label.config(image=None, text="Chưa xử lý")
        self.result_text.delete('1.0', tk.END)
        self.stats_label.config(text="Thống kê: Chưa có dữ liệu")
        self.status_label.config(text="Sẵn sàng")
        self.progress.stop()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingTextRecognitionApp(root)
    root.mainloop()