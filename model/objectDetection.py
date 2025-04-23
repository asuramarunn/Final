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

    def save_object_result(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, classes=self.classes)
            boxes = results[0].boxes

            person_index = 0
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].xyxy[0]
                conf = boxes[i].conf[0]
                if conf < 0.3:
                    continue

                # Cắt object từ frame
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                object_img = frame[y1:y2, x1:x2]

                # Lưu ảnh đã cắt
                filename = f"frame_{frame_index:06d}_person_{person_index:02d}.jpg"
                filepath = os.path.join(output_path, filename)
                cv2.imwrite(filepath, object_img)
                print(f"Đã lưu: {filepath}")
                person_index += 1

            frame_index += 1

        cap.release()

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


if __name__ == '__main__':
    model_path = 'YOLO11/weights/yolo11m.pt'
    # image_path = 'data/images/PedCrossing.jpg'
    # video_path = 'data/videos/video_test.mp4'
    # url = "rtsp://admin:Tinvl12345@@192.168.0.149:554/media/live/1/main"
    url = "rtsp://admin:Tinvl1903@@192.168.0.103:554/media/live/1/main"
    # url = "rtsp://admin:Tinvl12345@@192.168.1.13:554/media/live/1/main"
    # url = "rtsp://admin:Tinvl1903@@169.254.17.253:554/media/live/1/main"

    obj_detection = ObjectDetection(model_path)
    # obj_detection.show_image_result(image_path)
    # obj_detection.show_video_result(url)
    obj_detection.save_video_result(url)
    obj_detection.save_object_result(url, '../data/camera_data')
