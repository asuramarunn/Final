from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os


class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = [0]  # class human of

    def show_image_result(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(image_path, classes=self.classes)

        for i in range(len(results[0].boxes)):
            x1, y1, x2, y2 = results[0].boxes[i].xyxy[0]
            conf = results[0].boxes[i].conf[0]

            cv2.rectangle(img, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(img, f"{round(float(conf), 2)}", (int(x1), int(
                y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1920, 1080)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_video_result(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps,
                              (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1080, 720)
            cv2.resize(frame, (1080, 720))
            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            out.write(frame)

        cap.release()
        out.release()

    def save_object_result_from_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)  # Mở video từ file
        if not cap.isOpened():
            print(f"Lỗi khi mở video: {video_path}")
            return

        global frame_index  # Khai báo frame_index nếu chưa có
        if 'frame_index' not in globals():
            frame_index = 0  # Khởi tạo frame_index nếu chưa khởi tạo

        person_index = 0

        # Kiểm tra và tạo thư mục lưu trữ nếu chưa có
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Lặp qua từng frame trong video
        while True:
            ret, frame = cap.read()  # Đọc frame từ video
            if not ret:
                break  # Dừng khi không còn frame nào

            # Sử dụng model để phát hiện đối tượng
            results = self.model(frame, classes=self.classes)
            boxes = results[0].boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].xyxy[0]
                conf = boxes[i].conf[0]

                if conf < 0.2:  # Giảm ngưỡng độ tin cậy nếu cần
                    continue

                # Cắt object từ frame
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                object_img = frame[y1:y2, x1:x2]

                # Kiểm tra xem object_img có hợp lệ không
                if object_img is None or object_img.size == 0:
                    print(f"Lỗi: Không thể cắt ảnh cho frame {frame_index}")
                    continue

                # Tạo thư mục cho từng người nếu chưa có

                # Lưu ảnh đã cắt vào thư mục của người tương ứng
                filename = f"frame_{frame_index:06d}_{person_index}.jpg"
                filepath = os.path.join(output_path, filename)
                cv2.imwrite(filepath, object_img)
                print(f"Đã lưu: {filepath}")

                person_index += 1

            frame_index += 1

        cap.release()  # Giải phóng tài nguyên video sau khi xử lý xong

    def show_video_result(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps,
                              (frame_width, frame_height))

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % 1 == 0:
                results = self.model(frame, classes=self.classes)
                for i in range(len(results[0].boxes)):
                    x1, y1, x2, y2 = results[0].boxes[i].xyxy[0]
                    conf = results[0].boxes[i].conf[0]
                    if conf < 0.3:
                        continue

                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 255), 2)
                    cv2.putText(frame, f"{round(float(conf), 2)}", (int(x1), int(
                        y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 1080, 720)
                cv2.resize(frame, (1080, 720))
                cv2.imshow('image', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_num += 1
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def detect_and_crop(self, image, output_dir=None):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = self.model(image, classes=self.classes)
        boxes = results[0].boxes

        person_crops = []
        filtered_boxes = []
        crop_idx = 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            crop_img = image[y1:y2, x1:x2]
            if crop_img is None or crop_img.size == 0:
                continue

            person_crops.append({
                "image": crop_img,
                "box": (x1, y1, x2, y2),
                "conf": conf
            })
            filtered_boxes.append(box)  # chỉ lưu bbox đã lọc

            if output_dir:
                filename = f"crop_{crop_idx}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, crop_img)
                print(f"Saved: {filepath}")
                crop_idx += 1

        return filtered_boxes  # trả về bbox đã lọc

    def track_and_save_people(self, video_path, output_dir, conf_threshold=0.6, save_interval=5.0):
        """
        Track người trong video và lưu các đối tượng được phát hiện vào một thư mục duy nhất,
        chỉ lưu ảnh của cùng một người sau mỗi khoảng thời gian (mặc định 5 giây).
        
        Parameters:
        - video_path (str): Đường dẫn đến file video hoặc URL RTSP.
        - output_dir (str): Thư mục để lưu các hình ảnh cắt được.
        - conf_threshold (float): Ngưỡng độ tin cậy để lọc các đối tượng được phát hiện (mặc định: 0.6).
        - save_interval (float): Khoảng thời gian (giây) giữa các lần lưu ảnh cho cùng một track ID (mặc định: 5.0).
        """
        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video từ {video_path}")
            return

        # Lấy FPS của video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print("Lỗi: Không thể lấy FPS của video, sử dụng mặc định 30 FPS")
            fps = 30

        # Tính số frame tương ứng với save_interval
        frame_interval = int(fps * save_interval)

        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_index = 0
        total_objects = 0
        last_saved = {}  # Dictionary lưu frame_index lần lưu cuối cùng cho mỗi track_id

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sử dụng tracking của YOLOv11
            results = self.model.track(frame, classes=self.classes, persist=True, conf=conf_threshold)
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1  # Lấy track ID

                # Kiểm tra xem đã đủ khoảng thời gian để lưu ảnh mới cho track_id này chưa
                if track_id not in last_saved or (frame_index - last_saved.get(track_id, -frame_interval)) >= frame_interval:
                    # Cắt đối tượng từ frame
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img is None or crop_img.size == 0:
                        print(f"Lỗi: Không thể cắt ảnh tại frame {frame_index}, track ID {track_id}")
                        continue

                    # Lưu ảnh cắt được vào thư mục duy nhất
                    filename = f"person_{track_id}_{frame_index:06d}_1.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, crop_img)
                    print(f"Đã lưu: {filepath} (Track ID: {track_id})")
                    total_objects += 1

                    # Cập nhật frame_index lần lưu cuối cùng
                    last_saved[track_id] = frame_index

            frame_index += 1

        # Giải phóng tài nguyên
        cap.release()
        print(f"Hoàn tất xử lý video. Tổng số đối tượng lưu: {total_objects}")

if __name__ == '__main__':
    model_path = 'YOLO11/weights/yolo11m.pt'
    # image_path = 'data/images/PedCrossing.jpg'
    url = 'C:/Users/Acer/Documents/Zalo Received Files/gocCam1-5.mp4'
    # url = "rtsp://admin:Tinvl12345@@192.168.0.149:554/media/live/1/main"
    # url = "rtsp://admin:Tinvl12345@@192.168.1.13:554/media/live/1/main"
    # url = "rtsp://admin:Tinvl1903@@169.254.17.253:554/media/live/1/main"

    output_dir = "../data/cropped_persons"

    obj_detection = ObjectDetection(model_path)
    # obj_detection.show_image_result(image_path)
    # obj_detection.show_video_result(url)
    # obj_detection.save_video_result(url)
    # obj_detection.save_object_result_from_video(url, '../data/camera_data')
    obj_detection.track_and_save_people(url, output_dir, conf_threshold=0.7, save_interval=3.0)