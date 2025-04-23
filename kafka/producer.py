import sys, types
m = types.ModuleType('kafka.vendor.six.moves', 'Mock module')
setattr(m, 'range', range)
sys.modules['kafka.vendor.six.moves'] = m
import cv2
import threading
import time
import configparser
import sys
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
    print(f'Starting video stream from {rtsp_url}')
    
    video = cv2.VideoCapture(rtsp_url)
    if not video.isOpened():
        print(f"Failed to open stream: {rtsp_url}")
        return

    frame_count = 0
    frame_skip = 2
    while video.isOpened():
        success, frame = video.read()
        if not success:
            print(f"Failed to read frame from {rtsp_url}")
            break
    
        frame_count += 1
        
        # Process only every nth frame (where n = frame_skip)
        if frame_count % frame_skip != 0:
            continue
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (width, height))

        # Encode the frame for sending
        _, encoded_frame = cv2.imencode('.jpeg', resized_frame)
        frame_data = encoded_frame.tobytes()

        # Send the frame data
        future = producer.send(topic, frame_data)
        try:
            future.get(timeout=10)
        except KafkaError as e:
            print(f"Error sending frame to Kafka: {e}")
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
    num_cameras = int(config['settings']['num_cameras'])
    camera_urls = [url.strip() for url in config['settings']['camera_urls'].split('\n') if url.strip()]
    return num_cameras, camera_urls

if __name__ == '__main__':
    num_cameras, camera_urls = load_config()
    start_producers(camera_urls, num_producers=5, width=640, height=480)                                                                                                                                                                
                                                                                                                                