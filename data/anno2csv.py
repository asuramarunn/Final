import os
import pandas as pd

# Đường dẫn
# img_dir_lst = ["/Users/Acer/Downloads/PETA_dataset/3DPeS/archive", "/Users/Acer/Downloads/PETA_dataset/CUHK/archive"] 
# label_path_lst = ["/Users/Acer/Downloads/PETA_dataset/3DPeS/archive/Label.txt", "/Users/Acer/Downloads/PETA_dataset/CUHK/archive/Label.txt"]

img_dir_lst = ["/Users/Acer/Downloads/PETA_dataset/CAVIAR4REID/archive"] 
label_path_lst = ["/Users/Acer/Downloads/PETA_dataset/CAVIAR4REID/archive/Label.txt"]


# Danh sách các thuộc tính nhị phân
binary_attrs = [
    "accessoryHeadphone", "personalLess15", "personalLess30", "personalLess45", "personalLess60",
    "personalLarger60", "carryingBabyBuggy", "carryingBackpack", "hairBald", "footwearBoots",
    "lowerBodyCapri", "carryingOther", "carryingShoppingTro", "carryingUmbrella", "lowerBodyCasual",
    "upperBodyCasual", "personalFemale", "carryingFolder", "lowerBodyFormal", "upperBodyFormal",
    "accessoryHairBand", "accessoryHat", "lowerBodyHotPants", "upperBodyJacket", "lowerBodyJeans",
    "accessoryKerchief", "footwearLeatherShoes", "upperBodyLogo", "hairLong", "lowerBodyLongSkirt",
    "upperBodyLongSleeve", "lowerBodyPlaid", "lowerBodyThinStripes", "carryingLuggageCase",
    "personalMale", "carryingMessengerBag", "accessoryMuffler", "accessoryNothing", "carryingNothing",
    "upperBodyNoSleeve", "upperBodyPlaid", "carryingPlasticBags", "footwearSandals", "footwearShoes",
    "hairShort", "lowerBodyShorts", "upperBodyShortSleeve", "lowerBodyShortSkirt", "footwearSneaker",
    "footwearStocking", "upperBodyThinStripes", "upperBodySuit", "carryingSuitcase", "lowerBodySuits",
    "accessorySunglasses", "upperBodySweater", "upperBodyThickStripes", "lowerBodyTrousers",
    "upperBodyTshirt", "upperBodyOther", "upperBodyVNeck"
]

# Các thuộc tính đa lớp (màu)
colors = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow"]
multiclass_attrs = ["footwear", "hair", "lowerBody", "upperBody"]

# Tạo danh sách tất cả các cột
columns = ["image"] + binary_attrs
for attr in multiclass_attrs:
    for color in colors:
        columns.append(f"{attr}{color}")

# Khởi tạo danh sách records chung
all_records = []

for img_dir, label_path in zip(img_dir_lst, label_path_lst):    
    # Đọc label vào dict: {id: "att1 att2 ..."}
    id_to_attrs = {}
    with open(label_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                id_ = parts[0]
                attrs = parts[1:]
                id_to_attrs[id_] = attrs

    # Duyệt qua các ảnh
    for filename in os.listdir(img_dir):
        if filename.endswith(".bmp") or filename.endswith(".png") or filename.endswith(".jpg"):
            id_ = filename.split("_")[0]
            attrs = id_to_attrs.get(id_, [])

            # Tạo hàng mới với giá trị mặc định
            data_row = {col: 0 for col in columns}
            data_row["image"] = filename

            for label in attrs:
                if label in binary_attrs:
                    data_row[label] = 1
                elif label.startswith("upperBody") and label[9:] in colors:
                    data_row[f"upperBody{label[9:]}"] = 1
                elif label.startswith("lowerBody") and label[9:] in colors:
                    data_row[f"lowerBody{label[9:]}"] = 1
                elif label.startswith("hair") and label[4:] in colors:
                    data_row[f"hair{label[4:]}"] = 1
                elif label.startswith("footwear") and label[8:] in colors:
                    data_row[f"footwear{label[8:]}"] = 1

            all_records.append(data_row)

# Tạo DataFrame và lưu
df = pd.DataFrame(all_records, columns=columns)
df.to_csv("image_attributes_test.csv", index=False)
print("✅ CSV saved: image_attributes_test.csv")