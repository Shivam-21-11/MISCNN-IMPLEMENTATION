# Import some libraries
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.data_loading.interfaces import Image_interface
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, \
    dice_crossentropy, tversky_loss
from miscnn.processing.subfunctions import Resize, Normalization
import numpy as np


def calc_DSC(truth, pred, classes):
    dice_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate Dice
            dice = 2 * np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
            dice_scores.append(dice)
        except ZeroDivisionError:
            dice_scores.append(0.0)
    # Return computed Dice Similarity Coefficients
    return dice_scores


# Initialize Data IO & Image Interface
interface = Image_interface(classes=2, img_type="grayscale", img_format="tif")
data_path = "E:/PythonIDEP/CELLSEGMENTATION/DATA"
data_io = Data_IO(interface, data_path, delete_batchDir=True)

# Obtain the sample list
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Create a pixel value normalization Subfunction for z-score scaling
sf_zscore = Normalization(mode="z-score")
# Create a resizing Subfunction to shape 592x592
sf_resize = Resize((592, 592))

# Assemble Subfunction classes into a list
sf = [sf_resize]

# Initialize Preprocessor
pp = Preprocessor(data_io, batch_size=2, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="fullimage")

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, loss=tversky_crossentropy,
                       metrics=[tversky_loss, dice_soft, dice_crossentropy],
                       batch_queue_size=3, workers=5, learninig_rate=0.001)

from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard

# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5,
                          verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                          min_lr=0.00001)
cb_tb = TensorBoard(log_dir="tensorboard", histogram_freq=0, write_graph=True,
                    write_images=True)

# model.train(sample_list[1:], epochs=100, iterations=50, callbacks=[cb_lr, cb_tb])
# model.predict(sample_list[0:1])
#
# from IPython.display import display
# from PIL import Image
# import numpy as np
#
# # Load the first sample via MIScnn data loader
# sample_test = data_io.sample_loader(sample_list[0], load_seg=True, load_pred=True)
#
# # Visualize the ground truth segmentation
# seg_data = sample_test.seg_data * 100
# print("Shape of segmentation:", seg_data.shape)
# seg = Image.fromarray(np.reshape(seg_data, seg_data.shape[:-1]))
# display(seg)
#
# # Visualize the predicted segmentation
# pred_data = sample_test.pred_data * 100
# print("Shape of prediction:", pred_data.shape)
# pred = Image.fromarray(np.reshape(pred_data, pred_data.shape[:-1]))
# display(pred)
#
# dsc = calc_DSC(sample_test.seg_data, sample_test.pred_data, classes=2)
# print("DSC for Segmentation:", dsc[1])


from miscnn.evaluation.cross_validation import cross_validation
# Run cross-validation function
cross_validation(sample_list, model, k_fold=3, epochs=100, iterations=50,
                 evaluation_path="evaluation", draw_figures=True, callbacks=[cb_lr, cb_tb])
