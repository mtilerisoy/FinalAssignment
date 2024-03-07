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
import torch
import torch.nn as nn
import torch.optim as optim


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="5LSM0-FinalAssignment",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "Base U-Net Like",
    "dataset": "CityScapes",
    "epochs": 10,
    }
)


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # Define the transform
    resize_transform = transforms.Compose([
        transforms.ToTensor(),          # Convert to tensor
        transforms.Resize((256, 256)),  # Resize to 256x256
    ])

    # data loading
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',
                         transform=resize_transform, target_transform=resize_transform)
    
    # Print some information about the dataset and save to a file
    helpers.print_dataset_info(dataset)

    # visualize example images and labels
    helpers.visualize_dataset(dataset)  
    
    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # training/validation loop
    for epoch in range(wandb.config.epochs):
        running_loss = 0.0
        for inputs, masks in dataset:
            optimizer.zero_grad()
            outputs = model(inputs)
            masks = (masks * 255)
            loss = criterion(outputs, masks.long().squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataset)
        wandb.log({"loss": epoch_loss})

        # Save the model every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'models/version_1_epoch_{epoch}.pth')

        
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
