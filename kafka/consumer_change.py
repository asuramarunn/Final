import cv2
import time
import numpy as np
import os
import base64
from datetime import datetime
from flask import Flask, Response, jsonify, request, render_template
from kafka import KafkaConsumer
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from elasticsearch import Elasticsearch
from PIL import Image
import configparser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
import glob


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

import SigLIP as SigLIP
from SigLIP import InferDataset
from objectDetection import ObjectDetection


app = Flask(__name__, template_folder="../templates")
app.debug = True




last_frame_store = {}  # Lưu frame mới nhất theo camera_id
streaming_flags = {}   # Lưu trạng thái streaming của mỗi camera

labelToVietnamese = {
  'accessoryHeadphone': 'Tai nghe',
  'personalLess15': 'Dưới 15 tuổi',
  'personalLess30': 'Dưới 30 tuổi',
  'personalLess45': 'Dưới 45 tuổi',
  'personalLess60': 'Dưới 60 tuổi',
  'personalLarger60': 'Trên 60 tuổi',
  'carryingBackpack': 'Mang balo',
  'hairBald': 'Đầu hói',
  'footwearBoots': 'Giày bốt',
  'lowerBodyCapri': 'Quần capri',
  'carryingOther': 'Mang vật khác',
  'carryingShoppingTro': 'Mang xe đẩy mua sắm',
  'carryingUmbrella': 'Mang ô',
  'lowerBodyCasual': 'Quần thường',
  'upperBodyCasual': 'Áo thường',
  'personalFemale': 'Nữ',
  'carryingFolder': 'Mang cặp tài liệu',
  'lowerBodyFormal': 'Quần trang trọng',
  'upperBodyFormal': 'Áo trang trọng',
  'accessoryHairBand': 'Băng đô',
  'accessoryHat': 'Mũ',
  'lowerBodyHotPants': 'Quần ngắn nóng',
  'upperBodyJacket': 'Áo khoác',
  'lowerBodyJeans': 'Quần jeans',
  'accessoryKerchief': 'Khăn quàng',
  'footwearLeatherShoes': 'Giày da',
  'upperBodyLogo': 'Áo có logo',
  'hairLong': 'Tóc dài',
  'lowerBodyLongSkirt': 'Váy dài',
  'upperBodyLongSleeve': 'Áo dài tay',
  'lowerBodyPlaid': 'Quần kẻ caro',
  'lowerBodyThinStripes': 'Quần sọc mỏng',
  'carryingLuggageCase': 'Mang vali',
  'personalMale': 'Nam',
  'carryingMessengerBag': 'Mang túi đeo chéo',
  'accessoryMuffler': 'Khăn quàng cổ',
  'accessoryNothing': 'Không có phụ kiện',
  'carryingNothing': 'Không mang gì',
  'upperBodyNoSleeve': 'Áo không tay',
  'upperBodyPlaid': 'Áo kẻ caro',
  'carryingPlasticBags': 'Mang túi nhựa',
  'footwearSandals': 'Dép sandal',
  'footwearShoes': 'Giày',
  'hairShort': 'Tóc ngắn',
  'lowerBodyShorts': 'Quần ngắn',
  'upperBodyShortSleeve': 'Áo ngắn tay',
  'lowerBodyShortSkirt': 'Váy ngắn',
  'footwearSneaker': 'Giày thể thao',
  'footwearStocking': 'Tất',
  'upperBodyThinStripes': 'Áo sọc mỏng',
  'upperBodySuit': 'Áo vest',
  'carryingSuitcase': 'Mang vali kéo',
  'lowerBodySuits': 'Quần vest',
  'accessorySunglasses': 'Kính râm',
  'upperBodySweater': 'Áo len',
  'upperBodyThickStripes': 'Áo sọc dày',
  'lowerBodyTrousers': 'Quần dài',
  'upperBodyTshirt': 'Áo thun',
  'upperBodyOther': 'Áo khác',
  'upperBodyVNeck': 'Áo cổ V',
  'footwearBlack': 'Giày đen',
  'footwearBlue': 'Giày xanh dương',
  'footwearBrown': 'Giày nâu',
  'footwearGreen': 'Giày xanh lá',
  'footwearGrey': 'Giày xám',
  'footwearOrange': 'Giày cam',
  'footwearPink': 'Giày hồng',
  'footwearPurple': 'Giày tím',
  'footwearRed': 'Giày đỏ',
  'footwearWhite': 'Giày trắng',
  'footwearYellow': 'Giày vàng',
  'hairBlack': 'Tóc đen',
  'hairBlue': 'Tóc xanh dương',
  'hairBrown': 'Tóc nâu',
  'hairGreen': 'Tóc xanh lá',
  'hairGrey': 'Tóc xám',
  'hairOrange': 'Tóc cam',
  'hairPink': 'Tóc hồng',
  'hairPurple': 'Tóc tím',
  'hairRed': 'Tóc đỏ',
  'hairWhite': 'Tóc trắng',
  'hairYellow': 'Tóc vàng',
  'lowerBodyBlack': 'Quần đen',
  'lowerBodyBlue': 'Quần xanh dương',
  'lowerBodyBrown': 'Quần nâu',
  'lowerBodyGreen': 'Quần xanh lá',
  'lowerBodyGrey': 'Quần xám',
  'lowerBodyOrange': 'Quần cam',
  'lowerBodyPink': 'Quần hồng',
  'lowerBodyPurple': 'Quần tím',
  'lowerBodyRed': 'Quần đỏ',
  'lowerBodyWhite': 'Quần trắng',
  'lowerBodyYellow': 'Quần vàng',
  'upperBodyBlack': 'Áo đen',
  'upperBodyBlue': 'Áo xanh dương',
  'upperBodyBrown': 'Áo nâu',
  'upperBodyGreen': 'Áo xanh lá',
  'upperBodyGrey': 'Áo xám',
  'upperBodyOrange': 'Áo cam',
  'upperBodyPink': 'Áo hồng',
  'upperBodyPurple': 'Áo tím',
  'upperBodyRed': 'Áo đỏ',
  'upperBodyWhite': 'Áo trắng',
  'upperBodyYellow': 'Áo vàng'
}

# Tạo map ngược Vietnamese -> label
vietnameseToLabel = {v: k for k, v in labelToVietnamese.items()}



# ====== Load Model Model ======
human_detect = ObjectDetection("../weights/YOLO11/yolo11m.pt")
model_id = "google/siglip-so400m-patch14-384"
id2label = {0: 'accessoryHeadphone', 1: 'personalLess15', 2: 'personalLess30', 3: 'personalLess45', 4: 'personalLess60', 5: 'personalLarger60', 6: 'carryingBackpack', 7: 'hairBald', 8: 'footwearBoots', 9: 'lowerBodyCapri', 10: 'carryingOther', 11: 'carryingShoppingTro', 12: 'carryingUmbrella', 13: 'lowerBodyCasual', 14: 'upperBodyCasual', 15: 'personalFemale', 16: 'carryingFolder', 17: 'lowerBodyFormal', 18: 'upperBodyFormal', 19: 'accessoryHairBand', 20: 'accessoryHat', 21: 'lowerBodyHotPants', 22: 'upperBodyJacket', 23: 'lowerBodyJeans', 24: 'accessoryKerchief', 25: 'footwearLeatherShoes', 26: 'upperBodyLogo', 27: 'hairLong', 28: 'lowerBodyLongSkirt', 29: 'upperBodyLongSleeve', 30: 'lowerBodyPlaid', 31: 'lowerBodyThinStripes', 32: 'carryingLuggageCase', 33: 'personalMale', 34: 'carryingMessengerBag', 35: 'accessoryMuffler', 36: 'accessoryNothing', 37: 'carryingNothing', 38: 'upperBodyNoSleeve', 39: 'upperBodyPlaid', 40: 'carryingPlasticBags', 41: 'footwearSandals', 42: 'footwearShoes', 43: 'hairShort', 44: 'lowerBodyShorts', 45: 'upperBodyShortSleeve', 46: 'lowerBodyShortSkirt', 47: 'footwearSneaker', 48: 'footwearStocking', 49: 'upperBodyThinStripes', 50: 'upperBodySuit', 51: 'carryingSuitcase', 52: 'lowerBodySuits', 53: 'accessorySunglasses', 54: 'upperBodySweater', 55: 'upperBodyThickStripes', 56: 'lowerBodyTrousers', 57: 'upperBodyTshirt', 58: 'upperBodyOther', 59: 'upperBodyVNeck', 60: 'footwearBlack', 61: 'footwearBlue', 62: 'footwearBrown', 63: 'footwearGreen', 64: 'footwearGrey', 65: 'footwearOrange', 66: 'footwearPink', 67: 'footwearPurple', 68: 'footwearRed', 69: 'footwearWhite', 70: 'footwearYellow', 71: 'hairBlack', 72: 'hairBlue', 73: 'hairBrown', 74: 'hairGreen', 75: 'hairGrey', 76: 'hairOrange', 77: 'hairPink', 78: 'hairPurple', 79: 'hairRed', 80: 'hairWhite', 81: 'hairYellow', 82: 'lowerBodyBlack', 83: 'lowerBodyBlue', 84: 'lowerBodyBrown', 85: 'lowerBodyGreen', 86: 'lowerBodyGrey', 87: 'lowerBodyOrange', 88: 'lowerBodyPink', 89: 'lowerBodyPurple', 90: 'lowerBodyRed', 91: 'lowerBodyWhite', 92: 'lowerBodyYellow', 93: 'upperBodyBlack', 94: 'upperBodyBlue', 95: 'upperBodyBrown', 96: 'upperBodyGreen', 97: 'upperBodyGrey', 98: 'upperBodyOrange', 99: 'upperBodyPink', 100: 'upperBodyPurple', 101: 'upperBodyRed', 102: 'upperBodyWhite', 103: 'upperBodyYellow'}
ckpt_path = "../weights/SigLIP/best_model.pth"
device = "cpu"
Model = SigLIP.load_model(model_id, id2label, ckpt_path)
Model.eval()
size = 384
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])


# ====== Elasticsearch client ======
es = Elasticsearch("http://localhost:9200")

def save_to_elasticsearch(frame_info, doc_id=None):
    try:
        body = {
            "camera_id": frame_info['camera_id'],
            "timestamp": frame_info['timestamp'],
            "captions": frame_info['captions'],
            "fps": frame_info['fps'],
            "image_url": frame_info['image_url']
        }

        if doc_id:
            es.index(index="captions", id=doc_id, body=body)
        else:
            es.index(index="captions", body=body)

        print(f"[ES] ✅ Lưu vào Elasticsearch: {frame_info['image_url']}")
    except Exception as e:
        print(f"[ES] ❌ Lỗi: {e}")


def generate_frames(camera_id):
    topic = f'camera-stream-{camera_id}'
    consumer = KafkaConsumer(topic, bootstrap_servers='localhost:9092')

    streaming_flags[camera_id] = True

    for message in consumer:
        if not streaming_flags.get(camera_id, True):
            break

        frame = cv2.imdecode(np.frombuffer(message.value, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        last_frame_store[camera_id] = frame
        _, jpeg = cv2.imencode('.jpeg', frame)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n\r\n')


@app.route('/')
def index():
    # Đọc config
    config = configparser.ConfigParser()
    config.read('config.ini')

    num_cameras = int(config['settings']['num_cameras'])
    camera_urls_raw = config['settings']['camera_urls']
    camera_urls = [
        url.strip().replace('\\', '/')
        for url in camera_urls_raw.strip().splitlines()
        if url.strip()
    ]

    return render_template("index.html", num_cameras=num_cameras, camera_urls=camera_urls)

@app.route('/camera/<int:camera_id>')
def stream_camera(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_view/<int:camera_id>')
def camera_view(camera_id):
    streaming_flags[camera_id] = True  # luôn bật stream khi vào trang
    return render_template("camera_view.html", camera_id=camera_id)


@app.route('/analyze_captured/<int:camera_id>', methods=['POST'])
# def analyze(camera_id):
#     try:
#         streaming_flags[camera_id] = False  # Dừng stream

#         frame = last_frame_store.get(camera_id)
#         if frame is None:
#             return jsonify({"error": "No frame available"}), 404

#         # ==== 1. Lưu frame gốc ====
#         timestamp = datetime.now().isoformat()
#         timestamp_1 = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"camera{camera_id}_{timestamp_1}.jpg"
#         frame_dir = "static/frames"
#         os.makedirs(frame_dir, exist_ok=True)
#         save_path = os.path.join(frame_dir, filename)
#         cv2.imwrite(save_path, frame)

#         # Tạo thư mục crop trước
#         subfolder_name = f"{camera_id}_{timestamp_1}" 
#         crop_dir = os.path.join(frame_dir, "cropped_persons", subfolder_name)
#         os.makedirs(crop_dir, exist_ok=True)

#         # Gọi hàm detect_and_crop
#         boxes = human_detect.detect_and_crop(frame, output_dir=crop_dir)

#         # Đổi tên tất cả file crop_{idx}.jpg
#         crop_files = glob.glob(os.path.join(crop_dir, "crop_*.jpg"))
#         for idx, old_file_path in enumerate(sorted(crop_files)):  # Sắp xếp để đảm bảo thứ tự
#             old_filename = os.path.basename(old_file_path)
#             if old_filename.startswith("crop_"):
#                 new_filename = f"crop_{camera_id}_{timestamp_1}_{idx}.jpg"
#                 new_file_path = os.path.join(crop_dir, new_filename)
#                 print(f"Renaming {old_file_path} to {new_file_path}")
#                 os.rename(old_file_path, new_file_path)

#         # Tiếp tục xử lý dataset và captions
#         dataset = InferDataset(crop_dir, transform=transform)
#         cropped_files = []
#         captions = []
#         for idx, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop_filename = f"crop_{camera_id}_{timestamp_1}_{idx}.jpg"
#             crop_path = os.path.join(crop_dir, crop_filename)
            
#             cropped_files.append(f"/static/frames/cropped_persons/{subfolder_name}/{crop_filename}")

#             pixel_values, image_path = dataset[idx]
#             pred = SigLIP.infer_image(pixel_values, Model, device, threshold=0.8)
#             caption = [id2label[j] for j, p in enumerate(pred) if p == 1]


#             # === Sinh caption đặc trưng ===

#             color_bgr = [int(c) for c in np.random.randint(0, 256, size=3)]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness=2)
#             color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

#             captions.append({
#                 "caption": caption,
#                 "color": color_rgb
#             })


#             # === Lưu vào Elasticsearch ===
#             doc_id = f"{camera_id}_{timestamp_1}_{idx}"
#             save_to_elasticsearch({
#                 "camera_id": camera_id,
#                 "timestamp": timestamp,
#                 "fps": 0,
#                 "captions": caption,
#                 "image_url": f"/static/frames/cropped_persons/{subfolder_name}/{crop_filename}",
#             }, doc_id=doc_id)
        
#         # === Lưu lại frame đã vẽ box ===
#         cv2.imwrite(save_path, frame)


#         # ==== 3. Trả kết quả về frontend ====
#         return jsonify({
#             "status": "OK",
#             "image_url": f"/static/frames/{filename}",
#             "captions": captions,
#             "cropped_files": cropped_files
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

def analyze(camera_id):
    try:
        streaming_flags[camera_id] = False  # Dừng stream

        frame = last_frame_store.get(camera_id)
        if frame is None:
            return jsonify({"error": "No frame available"}), 404

        # ==== 1. Lưu frame gốc ====
        timestamp = datetime.now().isoformat()
        timestamp_1 = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera{camera_id}_{timestamp_1}.jpg"
        frame_dir = "static/frames"
        os.makedirs(frame_dir, exist_ok=True)
        save_path = os.path.join(frame_dir, filename)
        cv2.imwrite(save_path, frame)

        # ==== 2. Detect & crop ====
        subfolder_name = f"{camera_id}_{timestamp_1}" 
        crop_dir = os.path.join(frame_dir, "cropped_persons", subfolder_name)
        os.makedirs(crop_dir, exist_ok=True)

        boxes = human_detect.detect_and_crop(frame, output_dir=crop_dir)
        if not boxes:
            return jsonify({"status": "No people detected"}), 200

        # ==== 3. Đổi tên file crop (theo idx) ====
        cropped_files = []
        for idx in range(len(boxes)):
            old_path = os.path.join(crop_dir, f"crop_{idx}.jpg")
            new_filename = f"crop_{camera_id}_{timestamp_1}_{idx}.jpg"
            new_path = os.path.join(crop_dir, new_filename)
            os.rename(old_path, new_path)
            cropped_files.append(f"/static/frames/cropped_persons/{subfolder_name}/{new_filename}")

        # ==== 4. Caption hóa theo batch bằng DataLoader ====
        dataset = InferDataset(crop_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        captions = []
        idx_global = 0

        with torch.no_grad():
            for pixel_values, image_paths in dataloader:
                pixel_values = pixel_values.to(device)
                logits = Model(pixel_values)
                preds = (torch.sigmoid(logits.logits) > 0.8).int().cpu().numpy()

                for pred in preds:
                    caption = [id2label[j] for j, p in enumerate(pred) if p == 1]
                    box = boxes[idx_global]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Vẽ box + caption màu RGB
                    color_bgr = [int(c) for c in np.random.randint(0, 256, size=3)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness=2)
                    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

                    captions.append({
                        "caption": caption,
                        "color": color_rgb
                    })

                    # Lưu vào Elasticsearch
                    doc_id = f"{camera_id}_{timestamp_1}_{idx_global}"
                    save_to_elasticsearch({
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "fps": 0,
                        "captions": caption,
                        "image_url": cropped_files[idx_global],
                    }, doc_id=doc_id)

                    idx_global += 1

        # ==== 5. Ghi lại frame với box vẽ ====
        cv2.imwrite(save_path, frame)

        # ==== 6. Trả kết quả về frontend ====
        return jsonify({
            "status": "OK",
            "image_url": f"/static/frames/{filename}",
            "captions": captions,
            "cropped_files": cropped_files
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    searched = False

    if request.method == "POST":
        camera_id = request.form.get("camera_id")
        date = request.form.get("date")
        keyword = request.form.get("keyword", "").strip()

        keywords = []
        if keyword:
            # Tách keyword tiếng Việt, loại bỏ khoảng trắng
            keywords_vi = [kw.strip() for kw in keyword.split(",") if kw.strip()]
            # Chuyển tiếng Việt sang tiếng Anh (label)
            keywords = [vietnameseToLabel.get(kw, kw) for kw in keywords_vi]

        must_clauses = [{"match": {"camera_id": int(camera_id)}}]

        if date:
            must_clauses.append({
                "range": {
                    "timestamp": {
                        "gte": f"{date}T00:00:00",
                        "lte": f"{date}T23:59:59"
                    }
                }
            })

        if keywords:
            should_clauses = [{"match": {"captions": kw}} for kw in keywords]
            must_clauses.append({
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            })

        query = {
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "size": 50,
            "sort": [{"timestamp": "desc"}]
        }

        response = es.search(index="captions", body=query)
        hits = response["hits"]["hits"]

        for hit in hits:
            source = hit["_source"]
            img_url = source["image_url"].replace('\\', '/').lstrip('/')  # xóa dấu / đầu nếu có
            image_path = os.path.normpath(img_url)    # chuẩn hóa dấu phân tách thư mục

            print("Working directory:", os.getcwd())
            print("Đường dẫn đầy đủ:", image_path)
            print("File tồn tại?", os.path.exists(image_path))

            with open(image_path, "rb") as f:
                frame_bytes = f.read()
            frame_b64 = base64.b64encode(frame_bytes).decode()

            results.append({
                "camera_id": source["camera_id"],
                "timestamp": source["timestamp"],
                "fps": source["fps"],
                "captions": source["captions"],
                "frame": frame_b64
            })

        searched = True

    return render_template("search.html", results=results, searched=searched)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
