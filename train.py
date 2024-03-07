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


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".",       help="Path to the data")
    parser.add_argument("--model_version", type=str, default="1",   help="The version of the model")
    parser.add_argument("--device", type=str, default="cuda",       help="The device to train the model on")
    parser.add_argument("--learning_rate", type=float, default=0.01,help="The learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=10,           help="The number of epochs to train the model")
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="5LSM0-FinalAssignment",
        name = "snellius_training_run_"+args.model_version,
        
        # track hyperparameters and run metadata
        config={
        "version": args.model_version,
        "architecture": "Base U-Net Like",
        "dataset": "CityScapes",
        "device": args.device,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        }
    )

    # Define the transform
    resize_transform = transforms.Compose([
        transforms.ToTensor(),          # Convert to tensor
        transforms.Resize((256, 256)),  # Resize to 256x256
    ])

    # data loading
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',
                         transform=resize_transform, target_transform=resize_transform)
    
    # Create a DataLoader with batch size of 32
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    
    # Print some information about the dataset and save to a file
    helpers.print_dataset_info(dataset)

    # visualize example images and labels
    helpers.visualize_dataset(dataset)  
    
    # define model
    model = Model().to(args.device)

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # training/validation loop
    for epoch in range(wandb.config.epochs):
        start_time = time.time()  # Start time of the epoch
        running_loss = 0.0
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(args.device), masks.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(0))
            masks = (masks * 255)
            loss = criterion(outputs, masks.long().squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - start_time  # Time taken for the epoch
        epoch_loss = running_loss / len(dataloader)
        wandb.log({"loss": epoch_loss, "time (m)": epoch_time/60})

        # Save the model every 10 epochs
        if epoch+1 % 10 == 0:
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
