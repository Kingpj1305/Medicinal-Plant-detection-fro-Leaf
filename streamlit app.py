
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import h5py
# Load the pre-trained model
model = tf.keras.models.load_model(r'Model_resnet.h5')

# Define the class labels (modify as per your model)
class_labels = ['Alpinia Galanga (Rasna)','Amaranthus Viridis (Arive-Dantu)','Artocarpus Heterophyllus (Jackfruit)','Azadirachta Indica (Neem)','Basella Alba (Basale)','Brassica Juncea (Indian Mustard)','Ocimum Tenuiflorum (Tulsi)','Citrus Limon (Lemon)','Ficus Auriculata (Roxburgh fig)','Ficus Religiosa (Peepal Tree)','Hibiscus Rosa-sinensis','Jasminum (Jasmine)','Mangifera Indica (Mango)','Mentha (Mint)','Moringa Oleifera (Drumstick)','Muntingia Calabura (Jamaica Cherry-Gasagase)','Murraya Koenigii (Curry)','Nerium Oleander (Oleander)','Nyctanthes Arbor-tristis (Parijata)','Ocimum Tenuiflorum (Tulsi)','Piper Betle (Betel)','Plectranthus Amboinicus (Mexican Mint)','Pongamia Pinnata (Indian Beech)','Psidium Guajava (Guava)','Punica Granatum (Pomegranate)','Santalum Album (Sandalwood)','Syzygium Cumini (Jamun)','Syzygium Jambos (Rose Apple)','Tabernaemontana Divaricata (Crape Jasmine)','Trigonella Foenum-graecum (Fenugreek)']

# Streamlit app code
st.title('Image Classification of Leaves')
st.subheader('Upload an image for classification.')

# Image upload and prediction
uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    #st.write('')
    #st.write('Classifying...')

    # Preprocess the image
    image = image.resize((180, 180))
    #image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    #temp=class_labels[predicted_class]

    # Show the predicted class
    # st.write(f'Predicted Class: {class_labels[predicted_class]}')
    st.text("The prediction is :")
    pred = f'<p style="font-family:Lucida Console; color:#0ee38e; font-size: 40px;"><b>{class_labels[predicted_class]}</b></p>'
    st.markdown(pred, unsafe_allow_html=True)
    