# Alberto Zerbinati

import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import PeopleDataset
from model import PeopleDetectionCNN
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def train_model(save_model=False, model_path="models/people_detection_model.pt"):
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=7, fill=255
            ),
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize train/valid datasets and dataloaders
    train_path = "data/dataset/train"
    valid_path = "data/dataset/valid"

    train_dataset = PeopleDataset(train_path, transform=transform)
    valid_dataset = PeopleDataset(valid_path, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, drop_last=True)

    # Initialize the model
    model = PeopleDetectionCNN("cuda" if torch.cuda.is_available() else "cpu")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9, verbose=True)

    # Number of epochs
    n_epochs = 15

    # Training and Validation loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            output = model(data)
            output = output.cuda() if torch.cuda.is_available() else output
            target = target.cuda() if torch.cuda.is_available() else target
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Switch model to evaluation mode and perform validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm(valid_dataloader):
                output = model(data)
                output = output.cuda() if torch.cuda.is_available() else output
                target = target.cuda() if torch.cuda.is_available() else target
                loss = criterion(output, target)
                valid_loss += loss.item()

        scheduler.step()

        print(
            f"Epoch: {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_dataloader):.6f}, Valid Loss: {valid_loss/len(valid_dataloader):.6f}"
        )

    # Save the model
    if save_model:
        # Save for python usage
        torch.save(model.state_dict(), model_path + "h")

        # Reload the model from path
        model = PeopleDetectionCNN("cpu")
        model.load_state_dict(
            torch.load(model_path + "h", map_location=torch.device("cpu"))
        )

        model.eval()
        example = torch.ones(1, 3, 100, 100)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(model_path)

        # print the example output for reference with the cpp model
        print(model(example))

        print(f"Model saved to {model_path}")


def test_model(model):
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize test dataset and dataloader
    test_path = "data/dataset/test"
    test_dataset = PeopleDataset(test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # test loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            output = model(data)
            output = output.cuda() if torch.cuda.is_available() else output
            target = target.cuda() if torch.cuda.is_available() else target
            loss = criterion(output, target)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_dataloader):.6f}")


def save_annotated_images(model, save_path="output"):
    """
    just for debugging
    """
    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_path = "data/dataset/test"
    test_dataset = PeopleDataset(test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model.eval()

    correct = total = 0
    for idx, (model_image, model_label) in enumerate(test_dataloader):
        image = model_image[0]
        label = model_label.item()
        # Convert tensor image to numpy array and then to BGR for OpenCV
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # write the label as text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image, str(label), (10, 20), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA
        )

        # predict the label using the model
        output = model(model_image)
        _, predicted = torch.max(output.data, 1)

        color = (0, 255, 0) if predicted == label else (0, 0, 255)
        # write the predicted label as text on the image
        cv2.putText(
            image, str(predicted.item()), (10, 80), font, 0.3, color, 1, cv2.LINE_AA
        )

        # Save image
        file_name = f"{idx}.jpg"
        full_path = f"{save_path}/{file_name}"
        cv2.imwrite(full_path, image)

        if predicted == label:
            print(f"Correct prediction for image {idx}!")
            correct += 1
        else:
            print(f"Incorrect prediction for image {idx}!")
        total += 1

    print(f"Accuracy: {correct/total} ({correct}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_model", action="store_true", default=False)
    parser.add_argument("--save_trained_model", action="store_true", default=False)
    parser.add_argument("--evaluate_model", action="store_true", default=False)
    parser.add_argument(
        "--model_path", type=str, default="models/people_detection_model.pth"
    )

    args = parser.parse_args()

    if args.train_model:
        print(f"Training model (saving={args.save_trained_model})...")
        train_model(save_model=args.save_trained_model, model_path=args.model_path)

    if args.evaluate_model:
        print("Evaluating model...")
        # Load the model from path
        model = PeopleDetectionCNN("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.model_path + "h"))
        test_model(model=model)
        # save_annotated_images(model=model, save_path="output")
