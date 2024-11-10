'''

pip install mtcnn==0.1.0
pip installtensorflow==2.3.1
pip install keras==2.4.3
pip install keras-vggface==0.6
pip install keras_applications==1.0.8

'''



'''import os

actors = os.listdir('Data')

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filenames.append(os.path.join('data',actor,file))

print(len(filenames))

pickle.dump(filenames, open('filename.pkl','wb'))

'''
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filename.pkl','rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
# print(model.summary())

def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)

    result = model.predict(preprocessed_img).flatten()
    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

