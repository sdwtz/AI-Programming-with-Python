Meets Specifications

Great work!
Congratulations on completing your project!

I certainly enjoyed walking though your code. It's very clean and very well commented. I can clearly see the e"ort that has been put into this.
Very well implemented.
Here are some addition links that might help you
https://ikhlestov.github.io/pages/machine-learning/pytorch-notes/
https://arxiv.org/pdf/1609.08764.pdf
https://arxiv.org/ftp/arxiv/papers/1708/1708.03763.pdf



-Files Submitted
 The submission includes all required files. (Model checkpoints not required.)

-All the required files are included.
 Thanks for that, it helps to reproduce your results easily.



Part 1 - Development Notebook

-All the necessary packages and modules are imported in the first cell of the notebook
 All good here!
 Moving all the imports to the top is just a good practice as it helps in looking at the dependencies and the import requirements of the project in one go.
 Good work there taking care of the feedback from the previous reviewer into account. was by default in another cell. So, thanks for abiding by the rubric and moving it up with all the other imports.

-torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
 You have used the torchvision.transforms well in order to enhance the volume of the data by augmenting it with cropping, mirroring and rotation.
 Functional transforms give fine-grained control over the transformations. This is useful if you have to build a more complex transformation pipeline
 You can read more about it here.
 https://pytorch.org/docs/stable/torchvision/transforms.html

-The training, validation, and testing data is appropriately cropped and normalized
 You have resized and normalized the data in accordance with the input size accepted by the pretrained models.
 The following are code examples for showing how to use torchvision.transforms.CenterCrop()..
 https://www.programcreek.com/python/example/104835/torchvision.transforms.CenterCrop

-The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
 Good work.
 Each set is correctly loaded and you have chosen an appropriate batch size. You can also try trimming the batch size to 32.

-The data for each set is loaded with torchvision's DataLoader
 The code looks good here as well. Well done!
 Your code is well commented and clean, hence easy to go through.
 After loaded ImageFolder, we have to pass it to DataLoader. It takes a data set and returns batches of images and corresponding labels. Here we can set batch_size and shuffle (True/False) after each epoch. For this we need to pass data set, batch_size, shuffle into torch.utils.data.DataLoader() as below link
 https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2

-A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
 Good work loading the pretrained networks and using transfer learning.
 I saw that you have perfectly freezed the parameters of the pretrained layers so that there is not gradient computation on back propagation call, backward(), using below code
 for param in model.parameters():
 param.requires_grad = False
 It is only the final layers of our network, the layers that learn to identify classes specific to your project that need training.
 You cn learn more about it here
 https://towardsdatascience.com/how-do-pretrained-models-work-11fe2f64eaa2

-A new feedforward network is defined for use as a classifier using the features as input
 Nice work here creating a new classifier using nn.sequential

-The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
 This is perfect! You are only training the classifier layers and not the layers from the pretrained models. Good job.

-During training, the validation loss and accuracy are displayed
 Good work printing the logs.
 Both validation loss and accuracy are being captured in the logs.

-The network's accuracy is measured on the test data
 Great job on getting 80.59% accuracy on the testing set.

-There is a function that successfully loads a checkpoint and rebuilds the model
 There is a function that successfully loads a checkpoint and rebuilds the model. Good Job !!

-The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary
 Awesome
 You have successfully taken care of the importantance to capture the details of the epoch and the learning rate in the checkpoint.
 I can see that you have added both of the to the checkpoint

-The process_image function successfully converts a PIL image into an object that can be used as input to a trained model
 Good job taking care of the feedback. As earlier pointed out, you just need to make the image have a smaller side sized at 256 and maintain the aspect ratio.

-The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image
 The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image

-A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names
 This is awesome. You have done well with the inference and its display.


Part 2 - Command Line Application

-train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint
 Great job with the command line utility. This work smoothly.

-The training loss, validation loss, and validation accuracy are printed out as a network trains
 Awesome Training loss, validation loss etc. are being printed correctly during the training.

-The training script allows users to choose from at least two different architectures available from torchvision.models
 I see you are going with vgg16 is supported and available from torchvision.models. Good choice.

-The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
 Yes, it does and its works well.

-The training script allows users to choose training the model on a GPU
 Great implementation and good work using ‘cuda’ at all the places required.

-The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
 Everything seems perfect now.

-The predict.py script allows users to print out the top K classes along with associated probabilities
 Yes, it does! Tried it with varying values of K and it works like a charm.

-The predict.py script allows users to load a JSON file that maps the class values to other category names
 Yes the script has the capability to load from a JSON file.

-The predict.py script allows users to use the GPU to calculate the predictions
 Yes, it does and its works well.
