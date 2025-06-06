from objectDetection import ObjectDetection
from SigLIP import InferDataset, load_model, infer_image
import SigLIP as SigLIP
import cv2
import numpy as np
import random
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

frame = cv2.imread("../data/temp/frame.PNG")

ObjectDetection = ObjectDetection("../weights/YOLO11/yolo11m.pt")

def draw_multiline_text(img, text, org, font, font_scale, color, thickness, max_width):
    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (' ' if current_line != "" else '') + word
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w > max_width:
            if current_line != "":
                lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    x, y = org
    line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 10

    for i, line in enumerate(lines):
        y_line = y + i * line_height
        cv2.putText(img, line, (x, y_line), font, font_scale, color, thickness)

    return len(lines)  # 🟢 DÒNG NÀY GIÚP FIX LỖI


model_id = "google/siglip-so400m-patch14-384"
id2label = {0: 'accessoryHeadphone', 1: 'personalLess15', 2: 'personalLess30', 3: 'personalLess45', 4: 'personalLess60', 5: 'personalLarger60', 6: 'carryingBackpack', 7: 'hairBald', 8: 'footwearBoots', 9: 'lowerBodyCapri', 10: 'carryingOther', 11: 'carryingShoppingTro', 12: 'carryingUmbrella', 13: 'lowerBodyCasual', 14: 'upperBodyCasual', 15: 'personalFemale', 16: 'carryingFolder', 17: 'lowerBodyFormal', 18: 'upperBodyFormal', 19: 'accessoryHairBand', 20: 'accessoryHat', 21: 'lowerBodyHotPants', 22: 'upperBodyJacket', 23: 'lowerBodyJeans', 24: 'accessoryKerchief', 25: 'footwearLeatherShoes', 26: 'upperBodyLogo', 27: 'hairLong', 28: 'lowerBodyLongSkirt', 29: 'upperBodyLongSleeve', 30: 'lowerBodyPlaid', 31: 'lowerBodyThinStripes', 32: 'carryingLuggageCase', 33: 'personalMale', 34: 'carryingMessengerBag', 35: 'accessoryMuffler', 36: 'accessoryNothing', 37: 'carryingNothing', 38: 'upperBodyNoSleeve', 39: 'upperBodyPlaid', 40: 'carryingPlasticBags', 41: 'footwearSandals', 42: 'footwearShoes', 43: 'hairShort', 44: 'lowerBodyShorts', 45: 'upperBodyShortSleeve', 46: 'lowerBodyShortSkirt', 47: 'footwearSneaker', 48: 'footwearStocking', 49: 'upperBodyThinStripes', 50: 'upperBodySuit', 51: 'carryingSuitcase', 52: 'lowerBodySuits', 53: 'accessorySunglasses', 54: 'upperBodySweater', 55: 'upperBodyThickStripes', 56: 'lowerBodyTrousers', 57: 'upperBodyTshirt', 58: 'upperBodyOther', 59: 'upperBodyVNeck', 60: 'footwearBlack', 61: 'footwearBlue', 62: 'footwearBrown', 63: 'footwearGreen', 64: 'footwearGrey', 65: 'footwearOrange', 66: 'footwearPink', 67: 'footwearPurple', 68: 'footwearRed', 69: 'footwearWhite', 70: 'footwearYellow', 71: 'hairBlack', 72: 'hairBlue', 73: 'hairBrown', 74: 'hairGreen', 75: 'hairGrey', 76: 'hairOrange', 77: 'hairPink', 78: 'hairPurple', 79: 'hairRed', 80: 'hairWhite', 81: 'hairYellow', 82: 'lowerBodyBlack', 83: 'lowerBodyBlue', 84: 'lowerBodyBrown', 85: 'lowerBodyGreen', 86: 'lowerBodyGrey', 87: 'lowerBodyOrange', 88: 'lowerBodyPink', 89: 'lowerBodyPurple', 90: 'lowerBodyRed', 91: 'lowerBodyWhite', 92: 'lowerBodyYellow', 93: 'upperBodyBlack', 94: 'upperBodyBlue', 95: 'upperBodyBrown', 96: 'upperBodyGreen', 97: 'upperBodyGrey', 98: 'upperBodyOrange', 99: 'upperBodyPink', 100: 'upperBodyPurple', 101: 'upperBodyRed', 102: 'upperBodyWhite', 103: 'upperBodyYellow'}
ckpt_path = "../weights/SigLIP/best_model.pth"
Model = SigLIP.load_model(model_id, id2label, ckpt_path)
Model.eval()

boxes = ObjectDetection.detect_and_crop(frame, output_dir="../data/temp/cropped_persons") 

size = 384
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])



camera_id = 1
latest_frame = 1

colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in boxes]
captions = []
device = "cpu"




# Tạo folder lưu crop nếu chưa có
crop_folder = "../data/temp/cropped_persons"
os.makedirs(crop_folder, exist_ok=True)

dataset = InferDataset("../data/temp/cropped_persons", transform=transform)



print(f"Number of detected boxes: {len(boxes)}")
print(f"Number of images in dataset: {len(dataset)}")


# Chuyển boxes về numpy
xyxy_boxes = np.array([box.xyxy[0].cpu().numpy().astype(int) for box in boxes])

# Tạo bản sao của frame để vẽ
annotated_frame = frame.copy()

# Caption lưu lại từng người
captions = []

for i, (bbox, color) in enumerate(zip(boxes, colors)):
    # Gọi SigLIP
    pixel_values, image_path = dataset[i]
    pred = infer_image(pixel_values, Model, device, threshold=0.8)
    caption = [id2label[j] for j, p in enumerate(pred) if p == 1]
    text = ", ".join(caption)
    captions.append((text, color))

    # Vẽ box lên annotated_frame
    x1, y1, x2, y2 = xyxy_boxes[i]
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

# Tạo canvas mới: chiều cao như cũ, chiều rộng gấp đôi
h, w, _ = annotated_frame.shape
canvas = np.ones((h, w * 2, 3), dtype=np.uint8) * 255  # trắng nền

# Dán ảnh frame vào bên trái
canvas[:, :w] = annotated_frame
# Vẽ caption lên bên phải
max_caption_width = w - 40  # chiều rộng phần caption bên phải
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2

# Ước lượng chiều cao của 1 dòng văn bản
line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 10  # khoảng cách giữa dòng trong 1 caption
group_spacing = 20  # khoảng cách giữa các caption khác nhau

y_offset = 30  # điểm y bắt đầu vẽ
for i, (text, color) in enumerate(captions):
    org = (w + 20, y_offset)
    # Vẽ caption nhiều dòng
    n_lines = draw_multiline_text(canvas, f"Person {i+1}: {text}", org, font, font_scale, color, thickness, max_caption_width)
    
    # Cập nhật y_offset: số dòng * chiều cao + khoảng cách giữa caption tiếp theo
    y_offset += n_lines * line_height + group_spacing



# Lưu kết quả
cv2.imwrite("../data/temp/annotated_canvas.jpg", canvas)


# Lưu ảnh đã vẽ
output_path = "../data/temp"
cv2.imwrite(output_path, frame)
