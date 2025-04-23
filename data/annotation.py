import json

# Danh sách đầu vào mới (đã tinh chỉnh)
hair_color_words = ["blonde", "brown", "black", "red", "gray", "white", "auburn", "ginger", "brunette", "strawberry blonde", "platinum blonde"]
hair_style_words = ["long", "short", "curly", "straight", "wavy", "ponytail", "braided", "bun", "pigtails", "afro", "dreadlocks", "bangs"]
color_words = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "navy", "maroon", "gold", "silver", "bronze"]
upper_body_clothing = ["T-shirt", "Shirt", "Blouse", "Sweater", "Jacket", "Coat", "Hoodie", "Blazer", "Vest", "Tank top", "Cardigan", "Poncho", "Polo", "Tunic", "Crop top"]
lower_body_clothing = ["Pants", "Jeans", "Shorts", "Leggings", "Skirt", "Sweatpants"]
outerwear = ["Raincoat", "Windbreaker", "Overcoat", "Poncho", "Leather jacket"]
dresses = ["Dress", "Sundress"]
footwear = ["Shoes", "Boots", "Sneakers", "Sandals", "Slippers", "Heels", "Flip-flops"]
headwear = ["Hat", "Cap", "Beanie", "Helmet", "Headband", "Visor"]
accessories = ["Belt", "Scarf", "Gloves", "Tie", "Watch", "Bracelet", "Necklace", "Earrings", "Sunglasses", "Handbag", "Backpack", "Ring", "Socks"]
specialty_clothing = ["Suit", "Gown", "Uniform", "Kimono", "Robe", "Swimsuit", "Apron", "Costume"]
footwear_accessories = ["Shoelaces", "Stockings", "Tights"]
gender_age = ["Male", "Female", "Non-binary", "Boy", "Girl", "Man", "Woman", "Child", "Adult", "Senior", "Baby", "Teenager", "Elder"]

# Tạo dictionary với tất cả các tổ hợp, thêm "value": -1
character_attributes = {
    "Hair": [
        {"style": style, "color": color, "value": -1}
        for style in hair_style_words
        for color in hair_color_words
    ],
    "Upper Body Clothing": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in upper_body_clothing
        for color in color_words
    ],
    "Lower Body Clothing": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in lower_body_clothing
        for color in color_words
    ],
    "Outerwear": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in outerwear
        for color in color_words
    ],
    "Dresses": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in dresses
        for color in color_words
    ],
    "Footwear": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in footwear
        for color in color_words
    ],
    "Headwear": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in headwear
        for color in color_words
    ],
    "Accessories": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in accessories
        for color in color_words
    ],
    "Specialty Clothing": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in specialty_clothing
        for color in color_words
    ],
    "Footwear Accessories": [
        {"type": clothing, "color": color, "value": -1}
        for clothing in footwear_accessories
        for color in color_words
    ],
    "Gender and Age": [
        {"type": gender, "value": -1}
        for gender in gender_age
    ]
}

# Lưu dictionary vào file JSON
with open("character_attributes.json", "w", encoding="utf-8") as json_file:
    json.dump(character_attributes, json_file, indent=4, ensure_ascii=False)

print("Đã lưu dictionary vào file 'character_attributes.json'")