"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import wandb
import helpers
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
from torch.utils.data import random_split, ConcatDataset


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path",      type=str, default=".",      help="Path to the data")
    parser.add_argument("--model_version",  type=str, default="1",      help="The version of the model")
    parser.add_argument("--device",         type=str, default="cuda",   help="The device to train the model on")
    parser.add_argument("--learning_rate",  type=float, default=0.01,   help="The learning rate for the optimizer")
    parser.add_argument("--epochs",         type=int, default=10,       help="The number of epochs to train the model")
    parser.add_argument("--batch_size",     type=int, default=32,       help="The batch size to use in the data loaders")
    parser.add_argument("--cont",           type=bool, default=False,   help="Load a pre-trained model and continue training")
    parser.add_argument("--model_path",     type=str, default="models/model.pth",   help="path to pre-trained model")
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="5LSM0-FinalAssignment",
        name = "snellius-"+args.model_version,
        
        # track hyperparameters and run metadata
        config={
        "version": args.model_version,
        "architecture": "Base U-Net Like",
        "dataset": "CityScapes",
        "device": args.device,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
        }
    )

    # Define the transform
    resize_transform = transforms.Compose([
        transforms.ToTensor(),          # Convert to tensor
        transforms.Resize((256, 256)),  # Resize to 256x256
    ])

    # Define the transform for data augmentation
    rotation = helpers.RandomTransform(size=(256, 256), p=0.5, angle=60, jitter=False)
    color_jitter = helpers.RandomTransform(size=(256, 256), p=0.0, angle=30, jitter=True, brightness=0.3, contrast=[0.4,0.5], saturation=0.0, hue=0.10)

    # Load the training data
    original_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',
                         transform=resize_transform, target_transform=resize_transform)
    
    rotated_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', 
                               transforms=rotation)
    
    jittered_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', 
                               transforms=color_jitter)

    # Concatenate the original and augmented datasets
    train_dataset = ConcatDataset([original_dataset, rotated_dataset, jittered_dataset])

    # Define the size of the validation set
    val_size = int(0.2 * len(train_dataset))  # 20% for validation
    train_size = len(train_dataset) - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Print some information about the dataset and save to a file
    #helpers.print_dataset_info(dataset)

    # Visualize example images and labels
    #helpers.visualize_dataset(dataset)  

    # # Define model
    # if args.cont:
    #     model = torch.load(args.model_path)
    #     model = model.to(args.device)
    # else:
    #     model = Model().to(args.device)
    
    model = Model().to(args.device)

    # Define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # training/validation loop
    for epoch in range(wandb.config.epochs):
        # Start time of the epoch
        start_time = time.time()
        
        # Initialize the running loss
        running_loss = 0.0
        
        # Set the model to train mode
        model.train()
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(args.device), masks.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            masks = (masks * 255).long().squeeze()
            masks = utils.map_id_to_train_id(masks).to(args.device)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Set the model to evaluation mode
        model.eval()

        # Validation loop
        with torch.no_grad():
            val_running_loss = 0.0
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(args.device), masks.to(args.device)
                outputs = model(inputs)
                masks = (masks * 255).long().squeeze()
                masks = utils.map_id_to_train_id(masks).to(args.device)
                loss = criterion(outputs, masks)
                
                val_running_loss += loss.item()
        
        # Log the loss and time taken for the epoch
        epoch_time = time.time() - start_time  # Time taken for the epoch
        epoch_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)
        wandb.log({"Training Loss": epoch_loss,
                   "Validation Loss": val_loss,
                   "Time Took per Epoch (m)": epoch_time/60,
                   "Epoch": epoch})

        # Save the model every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"Model saved at models/{args.model_version}_{epoch}.pth")
            torch.save(model.state_dict(), f'models/{args.model_version}_{epoch}.pth')

        
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


    # save model


    # visualize some results

    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
