# Final Assignment

This repository contains the completed project for the 5LSM0 final assignment.
The project involved working with the Cityscapes dataset, training a deep neural network for semantic segmentation task,
and performing post-training improvements to increase the model efficiency.

This research explores the effects of model reduction
techniques, specifically quantization, and pruning, on the
efficiency and performance of semantic segmentation tasks. The
U-Net architecture, widely used in segmentation tasks, serves as
the base model. Evaluation is conducted using the CityScapes
dataset, focusing on two key benchmarks: peak performance
and efficiency. The study assesses metrics such as model size,
GFLOPs, and mean Dice score to measure performance and
efficiency.

## Getting Started

### Dependencies

We already created a DockerContainer with all dependencies to run on Snellius, in the run_main.sh file we refer to this container. You don't have to changes anything for this.

### Installing

To get started with this project, you need to clone the repository to Snellius or your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/mtilerisoy/MTI-5LSM0-FinalAssignment.git
```

After cloning the repository, navigate to the project directory:

```bash
cd MTI-5LSM0-FinalAssignment
```

Then, install the libraries:
```bash
pip install -r requirements.txt
```

### Dataset
- You can download [**cityscapes**](https://www.cityscapes-dataset.com/) dataset from [here](https://www.cityscapes-dataset.com/downloads/).
For this project, you have to download the following: [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4) and [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).


### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **run_container.sh:** Contains the script for running the container. In this file you have the option to enter your wandb keys if you have them and additional arguments if you have implemented them in the train.py file.


- **run_main:** Includes the code for building the Docker container. In this file, you only need to change the settings SBATCH (the time your job will run on the server) and ones you need to put your username at the specified location.


- **model.py:** Defines the neural network architecture.


- **model-UNet.py:** Contains the definition of the UNet architecture.


- **model-ENet.py:** Contains the definition of the ENet architecture.


- **train.py:** Contains the code for training the neural network.


- **helpers.py:** Contains helper functions to: print information and visualize the dataset, apply random transformation to the dataset, and a custom Dice Loss function to use in training.


- **utils.py:** Contains the utility function to map the class labels of training dataset to match test set.
[**See the repo**](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).


- **process_data.py:** Contains the functions for preprocessing images into PyTorch tensors suitable for model training, and postprocessing model predictions back into image format.


- **analyze.ipynb** This notebook has the necessary functions to load, visualize the segmentation, analyze the parameters, and process a pre-trained model.

## Command Line Arguments

The `train.py` script accepts the following command line arguments:

- `--data_path`: Path to the data. Default is current directory.
- `--model_version`: The version of the model. Default is "1". Used to log into wandb.
- `--device`: The device to train the model on. Default is "cuda".
- `--learning_rate`: The learning rate for the optimizer. Default is 0.01.
- `--epochs`: The number of epochs to train the model. Default is 10.
- `--batch_size`: The batch size to use in the data loaders. Default is 32.
- `--cont`: Boolean flag to indicate to load a pre-trained model and continue training. Default is False.
- `--model_path`: Path to pre-trained model (needed if cont=True). Default is "models/pretrained.pth".
- `--architecture`: Model Architecture. Default is "Base U-Net". Used to log into wandb.

### Training the Model
Use the following line to create a training job using a snellius environment:
```
sbatch run_main.sh
```
Use the following code to run the training.py on your local:
```
python3 train.py
```

If you want to train using different models, you have to change the **model.py**.
By default, this repo supports two models: UNet and ENet. You can simply switch between models by directly copying the
code from desired **model-XXX.py** script into **model.py**. 

### Authors

- M.T. Ilerisoy
- T.J.M. Jaspers
- C.H.B. Claessens
- C.H.J. Kusters


### To Cite This Repo

```bash
@misc{Efficiency-Improvements
  author = {Mustafa Talha Ilerisoy},
  title = {Efficiency Improvements of Deep Semantic Segmentation Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mtilerisoy/MTI-5LSM0-FinalAssignment}},
  commit = {master}
}
```
