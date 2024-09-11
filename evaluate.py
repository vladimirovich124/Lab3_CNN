import torch
import json
from model import ResNet50
from utils import setup_logger
from data import CustomDataset
from torch.utils.data import DataLoader
import os

def evaluate_model(config):
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    test_batch_dir = os.path.join(config.data_dir, "batches", "test_batch")
    test_label_file = os.path.join(test_batch_dir, "batch_registry_table.txt")
    test_dataset = CustomDataset(data_dir=test_batch_dir, label_file=test_label_file)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

if __name__ == "__main__":
    from config import Config
    config = Config()
    evaluate_model(config)
