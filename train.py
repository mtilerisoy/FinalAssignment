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
import random
import helpers

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
    print_dataset_info(dataset, filename)

    # visualize example images and labels
    for i in range(4):  # change this number to display more or fewer pairs
        image, label = dataset[i]
        
        # PyTorch tensors for images usually have the channels dimension first,
        # but matplotlib expects the channels dimension last.
        # So, we need to rearrange the dimensions using permute.
        image = image.permute(1, 2, 0)
        label = label.permute(1, 2, 0)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Adjust the size as needed
        ax[0].imshow(image)
        ax[0].set_title('Image')
        
        # Assuming the label is a PyTorch tensor, we need to convert it to numpy
        label = label.numpy()
        ax[1].imshow(label)
        ax[1].set_title('Label')

        # Save the figure
        plt.savefig(f'visualization/image_and_label_{i}.png')
        
        plt.show()    
    

    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)


    # training/validation loop

    # simulate a training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        
        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})
        
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
