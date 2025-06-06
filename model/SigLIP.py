import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import random
from torch.optim import AdamW
import traceback


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MultiLabelDataset(Dataset):
    def __init__(self, root, df, transform):
        self.root = root
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_path = os.path.join(self.root, item["image"])
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        labels = item[2:].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        return pixel_values, labels

    def __len__(self):
        return len(self.df)

class InferDataset(Dataset):
    def __init__(self, path, transform):
            self.path = path
            self.transform = transform

            self.jpg_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.jpg') or file.lower().endswith(".png") or file.lower().endswith(".PNG"):
                        full_path = os.path.join(root, file)
                        self.jpg_files.append(full_path)

    def __len__(self):
        return len(self.jpg_files)

    def __getitem__(self, idx):
    
        image_path = self.jpg_files[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        return pixel_values, image_path

# def data_infer_transform(data_path):
#     size = 384
#     mean = [0.5, 0.5, 0.5]
#     std = [0.5, 0.5, 0.5]

#     transform = Compose([
#         Resize((size, size)),
#         ToTensor(),
#         Normalize(mean=mean, std=std),
#     ])

#     dataset = InferDataset(data_path, transform=transform)

#     return dataset

def load_and_split_data(csv_path, root_path, processor, train_ratio=0.8, batch_size=2):
    df = pd.read_csv(csv_path)
    labels = list(df.columns)[2:]
    id2label = {id: label for id, label in enumerate(labels)}

    size = processor.size["height"]
    mean = processor.image_mean
    std = processor.image_std

    transform = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    dataset = MultiLabelDataset(root=root_path, df=df, transform=transform)
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size

    torch.manual_seed(42)
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

# def collate_fn(batch):
#     data = torch.stack([item[0] for item in batch])
#     target = torch.stack([item[1] for item in batch])
#     return data, target

# # train_dataloader = DataLoader(
# #     train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
# # )
# # val_dataloader = DataLoader(
# #     val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True
# # )

# #     return train_dataloader, val_dataloader, id2label

def load_model(model_id, id2label, ckpt_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, problem_type="multi_label_classification", id2label=id2label, ignore_mismatched_sizes=True
        )
    model = model.to(device)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

def evaluate(model, val_dataloader, device):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for pixel_values, labels in val_dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            losses.update(loss.item(), pixel_values.size(0))
    model.train()
    return losses.avg

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=30, lr=5e-5, patience=5):
    optimizer = AdamW(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_model_path = 'best_model.pth'
    epochs_no_improve = 0
    early_stop = False

    model.train()
    train_losses = AverageMeter()

    for epoch in range(num_epochs):
        train_losses.reset()
        for idx, (pixel_values, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values.to(device), labels=labels.to(device))
            loss = outputs.loss
            train_losses.update(loss.item(), pixel_values.size(0))
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'Epoch: [{epoch}]\tTrain Loss {train_losses.val:.4f} ({train_losses.avg:.4f})')

        val_loss = evaluate(model, val_dataloader, device)
        print(f'Epoch: [{epoch}]\tValidation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
            }, best_model_path)
            print(f'Saved best model at epoch {epoch} with validation loss: {best_loss:.4f}')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            early_stop = True
            break

    if not early_stop:
        print("Training completed without early stopping")
    print(f'Best model saved at: {best_model_path} with validation loss: {best_loss:.4f}')
    return best_model_path

def infer_image(pixel_values, model, device, threshold=0.5):
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values.unsqueeze(0))  # Thêm batch dimension
        logits = outputs.logits
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().cpu().numpy()[0]  # [num_labels]
    return preds

def infer_test_set(test_csv, test_root, model, processor, device, id2label, num_images=500, threshold=0.6):
    test_df = pd.read_csv(test_csv)
    test_dataset = MultiLabelDataset(root=test_root, df=test_df, transform=Compose([
        Resize((processor.size["height"], processor.size["height"])),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ]))

    random.seed(42)
    random_indices = random.sample(range(len(test_dataset)), num_images)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(random_indices)):
        fold_indices = [random_indices[i] for i in val_idx]
        results = []
        for idx in fold_indices:
            pixel_values, true_labels = test_dataset[idx]
            image_name = test_df.iloc[idx]['image']
            pred_labels = infer_image(pixel_values, model, device, threshold)
            pred_label_names = [id2label[i] for i, pred in enumerate(pred_labels) if pred == 1]
            true_label_names = [id2label[i] for i, true in enumerate(true_labels) if true == 1]
            results.append({
                'image': image_name,
                'predicted_labels': ', '.join(pred_label_names),
                'true_labels': ', '.join(true_label_names)
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv(f'inference_images_fold_{fold+1}.csv', index=False)
        print(f"Fold {fold+1} completed, saved {len(fold_indices)} images to inference_images_fold_{fold+1}.csv")

def evaluate_inference():
    def evaluate_image(pred_labels, true_labels):
        pred_set = set(pred_labels.split(', ')) if pred_labels else set()
        true_set = set(true_labels.split(', ')) if true_labels else set()
        correct = len(pred_set & true_set)
        extra = len(pred_set - true_set)
        missed = len(true_set - pred_set)
        precision = correct / (correct + extra) if (correct + extra) > 0 else 0
        recall = correct / (correct + missed) if (correct + missed) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return correct, extra, missed, precision, recall, f1

    for fold in range(1, 6):
        csv_path = f'inference_images_fold_{fold}.csv'
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"File {csv_path} not found. Skipping fold {fold}.")
            continue

        results = []
        for idx, row in df.iterrows():
            correct, extra, missed, precision, recall, f1 = evaluate_image(row['predicted_labels'], row['true_labels'])
            results.append({
                'image': row['image'],
                'correct': correct,
                'extra': extra,
                'missed': missed,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

        df_results = pd.DataFrame(results)
        avg_precision = df_results['precision'].mean()
        avg_recall = df_results['recall'].mean()
        avg_f1 = df_results['f1_score'].mean()

        print(f"\nFold {fold} Evaluation:")
        print(f"{'Image':<15} {'Correct':<8} {'Extra':<8} {'Missed':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        for idx, row in df_results.iterrows():
            print(f"{row['image']:<15} {row['correct']:<8} {row['extra']:<8} {row['missed']:<8} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1_score']:<10.4f}")
        print(f"\nFold {fold} Average Metrics:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")

        output_path = f'per_image_evaluation_fold_{fold}.csv'
        df_results.to_csv(output_path, index=False)
        print(f"\nFold {fold} evaluation results saved to {output_path}")

if __name__ == "__main__":
    
    model_id = "google/siglip-so400m-patch14-384"
    id2label = {0: 'accessoryHeadphone', 1: 'personalLess15', 2: 'personalLess30', 3: 'personalLess45', 4: 'personalLess60', 5: 'personalLarger60', 6: 'carryingBackpack', 7: 'hairBald', 8: 'footwearBoots', 9: 'lowerBodyCapri', 10: 'carryingOther', 11: 'carryingShoppingTro', 12: 'carryingUmbrella', 13: 'lowerBodyCasual', 14: 'upperBodyCasual', 15: 'personalFemale', 16: 'carryingFolder', 17: 'lowerBodyFormal', 18: 'upperBodyFormal', 19: 'accessoryHairBand', 20: 'accessoryHat', 21: 'lowerBodyHotPants', 22: 'upperBodyJacket', 23: 'lowerBodyJeans', 24: 'accessoryKerchief', 25: 'footwearLeatherShoes', 26: 'upperBodyLogo', 27: 'hairLong', 28: 'lowerBodyLongSkirt', 29: 'upperBodyLongSleeve', 30: 'lowerBodyPlaid', 31: 'lowerBodyThinStripes', 32: 'carryingLuggageCase', 33: 'personalMale', 34: 'carryingMessengerBag', 35: 'accessoryMuffler', 36: 'accessoryNothing', 37: 'carryingNothing', 38: 'upperBodyNoSleeve', 39: 'upperBodyPlaid', 40: 'carryingPlasticBags', 41: 'footwearSandals', 42: 'footwearShoes', 43: 'hairShort', 44: 'lowerBodyShorts', 45: 'upperBodyShortSleeve', 46: 'lowerBodyShortSkirt', 47: 'footwearSneaker', 48: 'footwearStocking', 49: 'upperBodyThinStripes', 50: 'upperBodySuit', 51: 'carryingSuitcase', 52: 'lowerBodySuits', 53: 'accessorySunglasses', 54: 'upperBodySweater', 55: 'upperBodyThickStripes', 56: 'lowerBodyTrousers', 57: 'upperBodyTshirt', 58: 'upperBodyOther', 59: 'upperBodyVNeck', 60: 'footwearBlack', 61: 'footwearBlue', 62: 'footwearBrown', 63: 'footwearGreen', 64: 'footwearGrey', 65: 'footwearOrange', 66: 'footwearPink', 67: 'footwearPurple', 68: 'footwearRed', 69: 'footwearWhite', 70: 'footwearYellow', 71: 'hairBlack', 72: 'hairBlue', 73: 'hairBrown', 74: 'hairGreen', 75: 'hairGrey', 76: 'hairOrange', 77: 'hairPink', 78: 'hairPurple', 79: 'hairRed', 80: 'hairWhite', 81: 'hairYellow', 82: 'lowerBodyBlack', 83: 'lowerBodyBlue', 84: 'lowerBodyBrown', 85: 'lowerBodyGreen', 86: 'lowerBodyGrey', 87: 'lowerBodyOrange', 88: 'lowerBodyPink', 89: 'lowerBodyPurple', 90: 'lowerBodyRed', 91: 'lowerBodyWhite', 92: 'lowerBodyYellow', 93: 'upperBodyBlack', 94: 'upperBodyBlue', 95: 'upperBodyBrown', 96: 'upperBodyGreen', 97: 'upperBodyGrey', 98: 'upperBodyOrange', 99: 'upperBodyPink', 100: 'upperBodyPurple', 101: 'upperBodyRed', 102: 'upperBodyWhite', 103: 'upperBodyYellow'}
    ckpt_path = "../weights/SigLIP/best_model.pth"

    size = 384
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    dataset = InferDataset("../data/cropped_persons", transform=transform)
    print(f"Number of images found: {len(dataset)}")

    device = "cpu"
    model = load_model(model_id, id2label, ckpt_path=ckpt_path)
    print("Model loaded!")

    try:
        model.eval()
        results = []

        for i in range(len(dataset)):
            pixel_values, image_path = dataset[i]

            pred = infer_image(pixel_values, model, device, threshold=0.8)
            pred_label_names = [id2label[j] for j, p in enumerate(pred) if p == 1]


            results.append({
                'image': image_path,
                'predicted_labels': ', '.join(pred_label_names)
            })

        # Lưu kết quả dự đoán vào file CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv('../data/cropped_persons/inference_person_2.csv', index=False)
        print(f"Saved inference results for {len(results)} images")

    except Exception as e:
        print("Error loading or running the model:", e)
        traceback.print_exc()
