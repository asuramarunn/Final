import sys, types
m = types.ModuleType('kafka.vendor.six.moves', 'Mock module')
setattr(m, 'range', range)
sys.modules['kafka.vendor.six.moves'] = m
from flask import Flask, Response, render_template, request, jsonify
from kafka import KafkaConsumer
import cv2
import numpy as np
import configparser
import time
import threading
import json
from datetime import datetime
from collections import deque
from human_att.code.florence2Class import Florence2Model

app = Flask(__name__)
message_counter = {}
model = None

frame_history = {}
MAX_FRAMES = 6

def load_model():
    global model
    model = Florence2Model()

last_frame_time = {}

def process_frame(frame_data, camera_id):
    frame_rgb = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    
    current_time = time.time()
    if camera_id not in last_frame_time:
        last_frame_time[camera_id] = current_time
        fps = 0.0
    else:
        time_diff = current_time - last_frame_time[camera_id]
        fps = 2 / time_diff if time_diff > 0 else 0.0
    last_frame_time[camera_id] = current_time
    
    processed_frame, captions = model.process_frame_tasks(frame_rgb)
    
    if captions:
        if camera_id not in frame_history:
            frame_history[camera_id] = deque(maxlen=MAX_FRAMES)
        
        timestamp = datetime.now().isoformat()
        _, jpeg = cv2.imencode('.jpeg', processed_frame)
        frame_info = {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'captions': captions,
            'frame': jpeg.tobytes(),
            'fps': fps  # Thêm FPS vào frame_info
        }
        frame_history[camera_id].appendleft(frame_info)
    
    cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return processed_frame

def kafkastream(consumer, topic, camera_id):
    global message_counter
    
    if camera_id not in message_counter:
        message_counter[camera_id] = 0
    
    for message in consumer:
        message_counter[camera_id] += 1
        
        if message_counter[camera_id] % 30 == 0:
            processed_frame = process_frame(message.value, camera_id)
            _, jpeg = cv2.imencode('.jpeg', processed_frame)
        else:
            frame = cv2.imdecode(np.frombuffer(message.value, np.uint8), cv2.IMREAD_COLOR)
            _, jpeg = cv2.imencode('.jpeg', frame)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def start_consumers(num_consumers, num_cameras):
    consumers = [KafkaConsumer(f'camera-stream-{i}', bootstrap_servers='localhost:9092') for i in range(num_cameras)]
    threads = []
    for idx in range(num_consumers):
        consumer = consumers[idx % num_cameras]
        topic = f'camera-stream-{idx}'
        thread = threading.Thread(target=kafkastream, args=(consumer, topic, idx))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    num_cameras = int(config['settings']['num_cameras'])
    return num_cameras

@app.route('/')
def index():
    num_cameras = load_config()
    return render_template('index.html', num_cameras=num_cameras)

@app.route('/frames')
def frames_page():
    num_cameras = load_config()
    return render_template('frames.html', num_cameras=num_cameras)

@app.route('/get_frames/<int:camera_id>')
def get_frames(camera_id):
    if camera_id not in frame_history:
        return jsonify([])
    
    frames_data = []
    for frame_info in list(frame_history[camera_id]):
        frames_data.append({
            'camera_id': frame_info['camera_id'],
            'timestamp': frame_info['timestamp'],
            'captions': frame_info['captions'],
            'fps': frame_info['fps'],
            'frame': frame_info['frame'].decode('latin1')
        })
    
    return jsonify(frames_data)

@app.route('/camera/<int:camera_id>')
def video_feed(camera_id):
    topic = f'camera-stream-{camera_id}'
    consumer = KafkaConsumer(topic, bootstrap_servers='localhost:9092')
    return Response(kafkastream(consumer, topic, camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/search_page')
def search_page():
    return render_template('search.html')

@app.route('/search')
def search_detections():
    query = {"query": {"bool": {"must": []}}}
    
    for key, value in request.args.items():
        if key.startswith('attribute'):
            index = key[9:]
            attribute = value
            value = request.args.get(f'value{index}')
            if attribute and value:
                query["query"]["bool"]["must"].append({"match": {f"attributes.{attribute}": value}})

    if not query["query"]["bool"]["must"]:
        return jsonify({"error": "No valid search criteria provided"}), 400

    results = es.search(index="person_detections", body=query)
    detections = [hit["_source"] for hit in results["hits"]["hits"]]

    return jsonify(detections)

if __name__ == '__main__':
    num_cameras = load_config()
    start_consumers(num_consumers=4, num_cameras=num_cameras)
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)