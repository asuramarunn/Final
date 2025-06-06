import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

import SigLIP as SigLIP


model_id = "google/siglip-so400m-patch14-384"
id2label = {0: 'accessoryHeadphone', 1: 'personalLess15', 2: 'personalLess30', 3: 'personalLess45', 4: 'personalLess60', 5: 'personalLarger60', 6: 'carryingBackpack', 7: 'hairBald', 8: 'footwearBoots', 9: 'lowerBodyCapri', 10: 'carryingOther', 11: 'carryingShoppingTro', 12: 'carryingUmbrella', 13: 'lowerBodyCasual', 14: 'upperBodyCasual', 15: 'personalFemale', 16: 'carryingFolder', 17: 'lowerBodyFormal', 18: 'upperBodyFormal', 19: 'accessoryHairBand', 20: 'accessoryHat', 21: 'lowerBodyHotPants', 22: 'upperBodyJacket', 23: 'lowerBodyJeans', 24: 'accessoryKerchief', 25: 'footwearLeatherShoes', 26: 'upperBodyLogo', 27: 'hairLong', 28: 'lowerBodyLongSkirt', 29: 'upperBodyLongSleeve', 30: 'lowerBodyPlaid', 31: 'lowerBodyThinStripes', 32: 'carryingLuggageCase', 33: 'personalMale', 34: 'carryingMessengerBag', 35: 'accessoryMuffler', 36: 'accessoryNothing', 37: 'carryingNothing', 38: 'upperBodyNoSleeve', 39: 'upperBodyPlaid', 40: 'carryingPlasticBags', 41: 'footwearSandals', 42: 'footwearShoes', 43: 'hairShort', 44: 'lowerBodyShorts', 45: 'upperBodyShortSleeve', 46: 'lowerBodyShortSkirt', 47: 'footwearSneaker', 48: 'footwearStocking', 49: 'upperBodyThinStripes', 50: 'upperBodySuit', 51: 'carryingSuitcase', 52: 'lowerBodySuits', 53: 'accessorySunglasses', 54: 'upperBodySweater', 55: 'upperBodyThickStripes', 56: 'lowerBodyTrousers', 57: 'upperBodyTshirt', 58: 'upperBodyOther', 59: 'upperBodyVNeck', 60: 'footwearBlack', 61: 'footwearBlue', 62: 'footwearBrown', 63: 'footwearGreen', 64: 'footwearGrey', 65: 'footwearOrange', 66: 'footwearPink', 67: 'footwearPurple', 68: 'footwearRed', 69: 'footwearWhite', 70: 'footwearYellow', 71: 'hairBlack', 72: 'hairBlue', 73: 'hairBrown', 74: 'hairGreen', 75: 'hairGrey', 76: 'hairOrange', 77: 'hairPink', 78: 'hairPurple', 79: 'hairRed', 80: 'hairWhite', 81: 'hairYellow', 82: 'lowerBodyBlack', 83: 'lowerBodyBlue', 84: 'lowerBodyBrown', 85: 'lowerBodyGreen', 86: 'lowerBodyGrey', 87: 'lowerBodyOrange', 88: 'lowerBodyPink', 89: 'lowerBodyPurple', 90: 'lowerBodyRed', 91: 'lowerBodyWhite', 92: 'lowerBodyYellow', 93: 'upperBodyBlack', 94: 'upperBodyBlue', 95: 'upperBodyBrown', 96: 'upperBodyGreen', 97: 'upperBodyGrey', 98: 'upperBodyOrange', 99: 'upperBodyPink', 100: 'upperBodyPurple', 101: 'upperBodyRed', 102: 'upperBodyWhite', 103: 'upperBodyYellow'}
ckpt_path = "../weights/SigLIP/best_model.pth"
device = "cpu"
Model = SigLIP.load_model(model_id, id2label, ckpt_path)
Model.eval()

dummy_input = torch.randn(1, 3, 384, 384)  # Kích thước ảnh đầu vào (phù hợp với Siglip)
torch.onnx.export(Model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])