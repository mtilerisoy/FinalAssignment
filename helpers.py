import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torch
import torch.nn as nn

def print_dataset_info(dataset, filename='output.txt'):
    """
    Prints information about the dataset to a file.

    Parameters:
    ----------
            dataset     : Pytorch Dataset object
            filename    : str
    
    Returns:
    --------
            None
    """
    
    with open(filename, 'w') as f:
        print(f"Type of dataset:                        {type(dataset)}", file=f)
        print(f"Length of dataset:                      {len(dataset)}", file=f)
        print(f"Number of classes:                      {len(dataset.classes)}", file=f)
        print(f"Shape of first image:                   {dataset[0][0].shape}", file=f)
        print(f"Shape of first semantic segmentation:   {dataset[0][1].shape}", file=f)

        img, smnt = dataset[0]
        print(f"Type of image and semantic seg:         {type(img)} | {type(smnt)}", file=f)
        print(f"Size of image and semantic seg:         {img.size()} | {smnt.size()}", file=f)
        print(f"Shape of image and semantic seg:        {img.shape} | {smnt.shape}", file=f)
        print(f"Data type of image and semantic seg:    {img.dtype} | {smnt.dtype}", file=f)
        print(f"Min and Max of image:                   {img.min()}, {img.max()}", file=f)
        print(f"Min and Max of semantic seg:            {smnt.min()}, {smnt.max()}", file=f)

        img255 = img * 255
        smnt255 = smnt * 255

        print(f"Min and Max of scaled image:            {img255.min()}, {img255.max()}", file=f)
        print(f"Min and Max of scaled semantic seg:     {smnt255.min()}, {smnt255.max()}", file=f)


def visualize_dataset(dataset, num_pairs=4):
    """
    Visualizes and SAVE a few images and their corresponding segmentation maps from the dataset.

    Parameters
    ----------
            dataset     : Pytorch Dataset object
            num_pairs   : int  (default: 4)

    Returns
    -------
            None
    """
    for i in range(num_pairs):
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


class RandomTransform:
    """
    A class used to apply random transformations to both an image and its corresponding target.

    Attributes
    ----------
            size : tuple
                the desired size after resizing the image and target
            p : float
                the probability of applying the random transformations

    Methods
    -------
            __call__(image, target)
                Applies the transformations to the image and target.
    """

    def __init__(self, size, p=0.5, angle=20, jitter=True, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1):
        """
        Construction for the class.

        Parameters
        ----------
                size    : tuple
                        the desired size after resizing the image and target
                p       : float
                        the probability of applying the random transformations
        """
        self.size = size
        self.p = p
        self.angle = angle
        self.jitter = jitter
        self.brightness = brightness #[brightness*0.5, brightness]
        self.contrast = contrast #[contrast*0.5, contrast]
        self.saturation = saturation #saturation*0.5, saturation]
        self.hue = hue

    def __call__(self, image, target):
        """
        Applies the transformations to the image and target.

        Parameters
        ----------
                image   : PIL Image
                        the image to be transformed
                target  : PIL Image
                        the target to be transformed

        Returns
        -------
                tuple   : the transformed image and target
        """
        
        # Resize
        resize = transforms.Resize(self.size)
        image = resize(image)
        target = resize(target)

        # Random horizontal flipping
        if random.random() < self.p:
            image = TF.hflip(image)
            target = TF.hflip(target)

        # Random vertical flipping
        if random.random() < self.p:
            image = TF.vflip(image)
            target = TF.vflip(target)

        # Random rotation
        angle = transforms.RandomRotation.get_params([-1*self.angle, self.angle])
        image = TF.rotate(image, angle)
        target = TF.rotate(target, angle)

        if self.jitter:
                # Color jitter
                color_jitter = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast,
                                                saturation=self.saturation, hue=self.hue)
                image = color_jitter(image)

        # Convert to tensor
        tensor_transform = transforms.ToTensor()
        image = tensor_transform(image)
        target = tensor_transform(target)

        return image, target


class MultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, log_loss=False):
        super(MultiClassDiceLoss, self).__init__()
        self.ingore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, input, target):
        smooth = 1.0

        # Apply softmax to input (model output)
        input = torch.softmax(input, dim=1)

        dice_loss = 0.0

        for class_index in range(input.size(1)):
            #valid = (target != self.ingore_index)
            valid = target.ne(self.ingore_index)
            # input_flat = input[:, class_index, :, :][valid].contiguous().view(-1)
            # target_flat = (target == class_index)[valid].contiguous().view(-1) # binary target for class_index
            input_flat = input[:, class_index, :, :][valid].view(-1)
            target_flat = (target == class_index)[valid].view(-1) # binary target for class_index

            intersection = (input_flat * target_flat).sum()

            dice_loss += 1 - ((2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth))
        
        
        mean_dice = dice_loss/input.size(1) # average loss over all classes

        if self.log_loss:
            mean_dice = -torch.log(mean_dice)

        return mean_dice