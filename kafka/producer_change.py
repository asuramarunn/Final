import cv2
import threading
import time
import configparser
from kafka import KafkaProducer
from kafka.errors import KafkaError
import numpy as np

def create_producer():
    return KafkaProducer(
        bootstrap_servers='localhost:9092',
        linger_ms=10,
        batch_size=16384,
        compression_type='gzip'
    )

def emit_video(rtsp_url, producer, topic, width=320, height=240):
    print(f'[INFO] Bắt đầu stream từ: {rtsp_url}')
    
    video = cv2.VideoCapture(int(rtsp_url) if rtsp_url.isdigit() else rtsp_url)
    
    if not video.isOpened():
        print(f"[ERROR] Không mở được stream: {rtsp_url}")
        return

    frame_count = 0
    frame_skip = 2
    while video.isOpened():
        success, frame = video.read()
        if not success:
            print(f"[ERROR] Không đọc được frame từ: {rtsp_url}")
            break
    
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        resized_frame = cv2.resize(frame, (width, height))
        _, encoded_frame = cv2.imencode('.jpeg', resized_frame)
        frame_data = encoded_frame.tobytes()

        try:
            producer.send(topic, frame_data).get(timeout=10)
        except KafkaError as e:
            print(f"[ERROR] Gửi frame lỗi: {e}")
            break

    video.release()

def start_producers(camera_urls, num_producers=2, width=320, height=240):
    producers = [create_producer() for _ in range(num_producers)]
    threads = []

    for idx, rtsp_url in enumerate(camera_urls):
        producer = producers[idx % num_producers]
        topic = f'camera-stream-{idx}'
        thread = threading.Thread(target=emit_video, args=(rtsp_url, producer, topic, width, height))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    urls = [line.strip() for line in config['settings']['camera_urls'].splitlines() if line.strip() and not line.strip().startswith(';')]
    return len(urls), urls

if __name__ == '__main__':
    num_cameras, camera_urls = load_config()
    start_producers(camera_urls, num_producers=5, width=640, height=480)
