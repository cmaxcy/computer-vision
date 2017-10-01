"""
Demonstration of use of ImageSet and TransferModel.

Google's InceptionV3 model has its learned weights frozen, barring the top layer
which is retrained. After this process is done, the top 249 layers are fine
tuned.

The data used is the Stanford Dogs Dataset which can be downloaded from
http://vision.stanford.edu/aditya86/ImageNetDogs/

It assumed that the dataset has been downloaded and stored in the same directory
with the name "Images".

Validation accuracy of 91.64% was achieved using these training parameters.
"""

from computer_vision import ImageSet, TransferModel
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD, Adam

# Split data into training and validation sets
ImageSet.partition_image_collection("Images", 10)

# Specify image locations and training augmentations
train_dir = "Train"
val_dir = "Validation"
train_aug = {"rescale": 1./255, "shear_range": .4, "zoom_range": .4,
    "horizontal_flip": True, "rotation_range": .4, "channel_shift_range": .4}
val_aug = {"rescale": 1./255}
resize_dims = (500, 500)

# Specify training hyperparameters
batch_size = 64
transfer_epochs = 3
fine_tune_epochs = 5
transfer_opt = Adam(lr=.001)
fine_tune_opt = Adam(lr=0.0001)
top_layers_to_unfreeze = 249

train_set = ImageSet(train_dir, train_aug, batch_size, resize_dims)
val_set = ImageSet(val_dir, aug_dict=val_aug, image_resize_dims=resize_dims)

# Prepare transfer-learning model
base_model = InceptionV3(weights="imagenet", include_top=False)
top_model_guide = [(1024, .5)]
model = TransferModel(base_model, top_model_guide)

# Fit model to data
model.fit(train_set, optimizers=(transfer_opt, fine_tune_opt), epochs=(transfer_epochs, fine_tune_epochs), val_set=val_set)
