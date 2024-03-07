import matplotlib.pyplot as plt


def print_dataset_info(dataset, filename='output.txt'):
    """
    Prints information about the dataset to a file.

    Parameters:
                dataset: Pytorch Dataset object
                filename: str
    
    Returns:
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

    Parameters:
                dataset: Pytorch Dataset object
                num_pairs: int  (default: 4)

    Returns:
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