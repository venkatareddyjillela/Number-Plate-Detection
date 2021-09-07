import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageOps


@st.cache(allow_output_mutation=True)

def load_model():
    model = cv2.dnn.readNet('yolov3_weights\yolov3_training_last.weights', 'yolov3_cfg\yolov3_testing.cfg')
    return model
with st.spinner("model is being loaded"):
    model = load_model()

st.write("""
         # Number Plate Detection
         """
         )

file = st.file_uploader("please upload image", type = ["jpg"])
st.set_option('deprecation.showfileUploaderEncoding', False)
def remove(string):
    return "".join(string.split())

def import_and_predict(img,model):
    # size = (50,50)    
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # image = np.asarray(image)
    classes = []
    with open('classes\classes.txt', "r") as f:
        classes = f.read().splitlines()
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    # img = cv2.imread(image)
    img = np.asarray(img)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers_names = model.getUnconnectedOutLayersNames()
    layerOutputs = model.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            imCrop = img[int(y):int(y+h), int(x):int(x+w)]
            reader = easyocr.Reader(["en"], gpu=False)
            output = reader.readtext(imCrop)
            if output != []:
                print(remove(output[0][1]))
                cv2.rectangle(img, (0, 0), (130, 30), (255, 255, 255), -1)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(
                    img, "NP:" + remove(str(output[0][1])), (2, 13), font, 1, (255, 0, 0), 1)
                cv2.putText(img, "Probability:" + remove(
                    str(round(output[0][2], 2)*100)), (1, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    return img, output[0][1],round(output[0][2], 2)*100


if file is None:
    st.text("Please upload an image file")
else:
    image =Image.open(file)
    # sr = image.tostring()
    # st.image(image, use_column_width=True)
    modified_image,predictions,probability = import_and_predict(image, model)
    
    
    st.write("""
         ## Image with detection of number plate
         """
         )
    st.image(modified_image)
    st.write("""
         ## Detected Plate Number: ##
         """
         )
    st.write(predictions)
    st.write("""
          Detected Plate Number Probability**(%)**:
         """
         )
    st.write(probability) 
