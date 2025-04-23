import json
import spacy
import pandas as pd

# Tải mô hình spaCy
nlp = spacy.load("en_core_web_sm")

# Ánh xạ màu sắc (có thể mở rộng thêm)
color_mapping = {
    "khaki": "brown",
    "navy": "blue",
    "maroon": "red",
    # Thêm các từ đồng nghĩa khác nếu cần
}

# Đọc file JSON chứa danh sách attribute
def load_attributes(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Hàm tìm cụm danh từ và tính từ bổ nghĩa
def extract_attributes(doc):
    attributes = []
    for token in doc:
        if token.pos_ == "NOUN":  # Tìm danh từ (shirt, pants, hair, v.v.)
            noun = token.text
            color = None
            style = None
            # Duyệt các token con để tìm tính từ bổ nghĩa
            for child in token.children:
                if child.dep_ in ("amod", "compound") and child.pos_ in ("ADJ", "NOUN"):
                    if child.text in color_mapping:
                        color = color_mapping[child.text]
                    elif child.text in ["red", "blue", "green", "yellow", "purple", "orange", "pink", 
                                      "brown", "black", "white", "gray", "navy", "maroon", 
                                      "gold", "silver", "bronze", "platinum", "strawberry", "auburn", "ginger"]:
                        color = child.text
                    else:
                        style = child.text
            if color or style:
                attributes.append({"noun": noun, "color": color, "style": style})
    return attributes

# Hàm xử lý caption với spaCy
def process_caption(caption, attributes):
    result = attributes.copy()
    doc = nlp(caption.lower())
    
    # Trích xuất các thuộc tính từ caption
    extracted_attrs = extract_attributes(doc)
    
    # Xử lý Hair
    for hair in result["Hair"]:
        for attr in extracted_attrs:
            if attr["noun"] == "hair" and attr["style"] == hair["style"] and attr["color"] == hair["color"]:
                hair["value"] = 1
    
    # Xử lý Upper Body Clothing
    for clothing in result["Upper Body Clothing"]:
        for attr in extracted_attrs:
            if attr["noun"] == clothing["type"].lower() and attr["color"] == clothing["color"]:
                clothing["value"] = 1
    
    # Xử lý Lower Body Clothing
    for clothing in result["Lower Body Clothing"]:
        for attr in extracted_attrs:
            if attr["noun"] == clothing["type"].lower() and attr["color"] == clothing["color"]:
                clothing["value"] = 1
    
    # Xử lý Gender and Age (dùng kiểm tra đơn giản vì không cần màu)
    for gender_age in result["Gender and Age"]:
        if gender_age["type"].lower() in caption.lower():
            gender_age["value"] = 1
    
    # Các danh mục khác (Outerwear, Dresses, Footwear, v.v.) có thể thêm tương tự
    
    return result

# Hàm lưu kết quả
def save_result(result, output_file):
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

# Ví dụ sử dụng
if __name__ == "__main__":
    json_file = "attributes.json"  # Thay bằng đường dẫn thực tế
    attributes = load_attributes(json_file)
    

    df = pd.read_csv("code/data/valid_entries.csv")

    print(df["human_caption"].count())


    for caption in df["human_caption"]:
        

        result = process_caption(caption, attributes)
        save_result(result, "output.json")
    
    print("Done")