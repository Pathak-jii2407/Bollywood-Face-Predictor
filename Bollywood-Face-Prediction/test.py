# Recommend that image

import cv2
import pickle
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from PIL import Image


feature_list = np.array(pickle.load(open('embedding.pkl','rb')))

filenames = pickle.load(open('filename.pkl','rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='max')

detector = MTCNN()

# load image -> face detection and extract features
sample_img = cv2.imread('sample/srkfake.jpeg')

results = detector.detect_faces(sample_img)


x, y, width, height = results[0]['box']
x = max(0, x)
y = max(0, y)
width = min(width, sample_img.shape[1] - x)
height = min(height, sample_img.shape[0] - y)


face = sample_img[y:y+height, x:x+width]

# cv2.imshow('Output',face)
#
#
# cv2.waitKey(0)
#


# extract the features
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')

expanded_image = np.expand_dims(face_array, axis=0)

preprocessed_img = preprocess_input(expanded_image)

result = model.predict(preprocessed_img).flatten()

# print(result)
# print(result.shape)

# find cosine distance of the current image with all the 8655 features
similarity=[]
for i in range(len(feature_list)):
    # print(cosine_similarity(result.reshape(1, -1), feature_list[0].reshape(1, -1))[0,0])
     similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]
print(index_pos)

temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('ootput', temp_img)
cv2.waitKey(0)








