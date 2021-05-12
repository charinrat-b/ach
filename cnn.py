import os
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.layers import ConvLSTM2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
import pywt
from tqdm import tqdm
from PIL import Image
import ahocorasick
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from ahocorasick import Automaton

tree = Automaton()
tree.add_word(str(0), str(image.tobytes()))
for x,y in enumerate(image.tobytes()):
    tree.add_word(str(x), y)

def convert_image_grayscale_dpi(filename):
    image = Image.open(filename).convert("L").resize((500,500))
    image.save(filename, dpi=(200,200))
    
train_path = "dataset/signature/train"

# convert original image (x,y) from dataframe to grayscale, dpi=200, waveform
for folder in tqdm(os.listdir(train_path)):
    for file in os.listdir(train_path+folder+"/"):
        convert_image_grayscale_dpi(train_path+folder+"/"+file)
        original = Image.open(train_path+folder+"/"+file)
        coeffs2 = pywt.dwt2(original, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        plt.imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([]);
        plt.savefig(train_path+folder+"/"+file)
        
pattern_found, folder_name = [], []
for folder in tqdm(os.listdir("dataset/signature/train")):
    # store each folder
    folder_name.append(folder)
    # load image number one from each folder as assign
    ori = Image.open("dataset/signature/train"+folder+"/1.png")
    # initialize structure tree aho-corasick
    tree = Automaton()
    # iterate each column pixel and convert to byte for assign as pattern
    for x in range(ori.size[1]):
        index = str(x)
        # convert array of value [255,255,255,255] to bytes
        value = str(np.asarray(ori)[x].tobytes())
        # assign to structure
        tree.add_word(index, value)
    for x in range(2,11):
        #  find similar pattern base assign pattern
        similarity = 0
        # load image, image already grayscale & dpi already 200
        test = Image.open(f"dataset/signature/train/{folder}/{x}.png")
        # convert array of image to bytes and find similarity
        for x in range(test.size[1]):
            index = str(x)
            value = str(np.asarray(test)[x].tobytes())
            if tree.get(index, value) != None :
                # found will increase base their similarity
                similarity += 1
            else :
                pass
        # append our found similarity for create histogram later
        pattern_found.append(similarity)
        # clear ahocorasick pattern for new pattern
        Automaton.clear(tree)
history = pd.DataFrame(columns=[f"PATTERN {X}" for X in range(2,11)])
history["FOLDER"] = folder_name
history = history.drop("FOLDER",axis=1).fillna(288).astype(int)
history["FOLDER"] = folder_name

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

# loop over the image paths
for imagePath in tqdm(os.listdir("dataset/signature/train/An Xiaoxiao")):
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread("dataset/signature/train/An Xiaoxiao/"+imagePath)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[filename] = hist
    
# METHOD #1: UTILIZING OPENCV
# initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))
# loop over the comparison methods
for methodName, method in OPENCV_METHODS:
    # initialize the results dictionary and the sort
    # direction
    results = {}
    reverse = False
    # if we are using the correlation or intersection
    # method, then sort the results in reverse order
    if methodName in ("Correlation", "Intersection"):
        reverse = True
    # loop over the index
    for (k, hist) in index.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = cv2.compareHist(index["2.png"], hist, method)
        results[k] = d
    # sort the results
    results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
        # show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images["1.png"])
    plt.axis("off")
    # initialize the results figure
    fig = plt.figure("Results: %s" % (methodName), figsize=(20,20))
    fig.suptitle(methodName, fontsize = 20)
    # loop over the results
    for (i, (v, k)) in enumerate(results):
        # show the result
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        plt.imshow(images[k])
        plt.axis("off")
# show the OpenCV methods
plt.show()

EPOCHS = 50
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root ="dataset/signature".replace("\\","/")
width=256
height=256
depth=3

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
        
 image_list, label_list = [], []
try :
    root_dir = os.listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for folder in root_dir :
        folder_list = os.listdir(f"{directory_root}/{folder}")
except Exception:
    pass
    
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    for folder in tqdm(os.listdir(directory_root)):
        for file in os.listdir(directory_root+"/"+folder+"/"):
            curr_path = directory_root+"/"+folder+"/"+file
            image_list.append(convert_image_to_array(curr_path))
            label_list.append(folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
    
    
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
n_classes = len(label_binarizer.classes_)
image_size = len(image_list)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.4, random_state = 1) 
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
    
 inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (8, 8), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )
    
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
