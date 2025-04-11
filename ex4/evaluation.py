import torch
import os
from PIL import Image

def save_results(model, test_dir, transform, output_file="pred.csv", device="cpu"):
    model.eval()
    device = torch.device(device)
    model.to(device)

    predictions = []
    filenames = []

    for file in sorted(os.listdir(test_dir)):
        file_path = os.path.join(test_dir, file)
        if os.path.isfile(file_path):
            try:
                image = Image.open(file_path).convert("RGB")
            except Exception as e:
                print(f"Skipping file {file} due to error: {e}")
                continue

            image = transform(image)
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                pred = torch.argmax(output, dim=1).item()

            filenames.append(file)
            predictions.append(pred)

    with open(output_file, "w") as f:
        for fname, pred in zip(filenames, predictions):
            f.write(f"{fname},{pred}\n")

    print(f"Saved {len(predictions)} predictions to {output_file}")