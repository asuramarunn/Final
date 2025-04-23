import json

# Đọc file JSON
def load_attributes(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Tạo danh sách chuỗi theo dạng [color] [noun] hoặc [type]
def create_attribute_strings(json_file):
    attributes = load_attributes(json_file)
    texts = []
    
    # Xử lý Hair
    for hair in attributes.get("Hair", []):
        text = f"{hair['color']} {hair['style']} hair"
        texts.append(text)
    
    # Xử lý Upper Body Clothing
    for clothing in attributes.get("Upper Body Clothing", []):
        text = f"{clothing['color']} {clothing['type'].lower()}"
        texts.append(text)
    
    # Xử lý Lower Body Clothing
    for clothing in attributes.get("Lower Body Clothing", []):
        text = f"{clothing['color']} {clothing['type'].lower()}"
        texts.append(text)
    
    # Xử lý Outerwear
    for outerwear in attributes.get("Outerwear", []):
        text = f"{outerwear['color']} {outerwear['type'].lower()}"
        texts.append(text)
    
    # Xử lý Dresses
    for dress in attributes.get("Dresses", []):
        text = f"{dress['color']} {dress['type'].lower()}"
        texts.append(text)
    
    # Xử lý Footwear
    for footwear in attributes.get("Footwear", []):
        text = f"{footwear['color']} {footwear['type'].lower()}"
        texts.append(text)
    
    # Xử lý Headwear
    for headwear in attributes.get("Headwear", []):
        text = f"{headwear['color']} {headwear['type'].lower()}"
        texts.append(text)
    
    # Xử lý Specialty Clothing
    for specialty in attributes.get("Specialty Clothing", []):
        text = f"{specialty['color']} {specialty['type'].lower()}"
        texts.append(text)
    
    # Xử lý Gender and Age
    for gender_age in attributes.get("Gender and Age", []):
        text = f"{gender_age['type'].lower()}"
        texts.append(text)
    
    return texts

# Ví dụ sử dụng
if __name__ == "__main__":
    json_file = "code/data/character_attributes.json"  # Thay bằng đường dẫn thực tế
    texts = create_attribute_strings(json_file)
    print("texts =", texts)