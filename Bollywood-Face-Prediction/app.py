import streamlit as st
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
import pickle

# Initialize the face detector and VGGFace model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed features and filenames
feature_list = pickle.load(open(f"{os.path.join('embedding.pkl')}", 'rb'))
filenames = pickle.load(open(f"{os.path.join('filename.pkl')}", 'rb'))

# Save the uploaded image function
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join(f"{os.path.join('uploads')}", uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving the image: {e}")
        return False

# Extract features from the image using the model
def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:  # No face detected
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    # Convert to 224x224 pixels and preprocess
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    # Extract features using VGGFace model
    result = model.predict(preprocessed_img).flatten()
    return result

# Recommend a celebrity based on cosine similarity
def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# Streamlit app UI
st.title('Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Save the uploaded image
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)

        # Extract features from the image
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)

        if features is not None:
            # Recommend a celebrity
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

            # Display the uploaded and recommended celebrity image
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your uploaded image')
                st.image(display_image, width=100)
            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos], width=100)
        else:
            st.error("No face detected in the uploaded image. Please try again with a different image.")