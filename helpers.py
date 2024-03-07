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