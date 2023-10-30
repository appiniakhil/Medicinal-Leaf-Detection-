from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tensorflow as tf
import logging
import sys
from PIL import Image
import cv2
import numpy as np
import io
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import re
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from leaf_composition import comp



IMAGE_SIZE = 224

# Loading model
model = tf.keras.models.load_model('leaf_classification_model2.h5')  # Replace with the path to your trained model file

model.make_predict_function()

# Define the UPLOAD_FOLDER as a relative path
UPLOAD_FOLDER = 'upload'

# Use the built-in Flask static folder for serving static files (CSS, JavaScript, etc.)
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define class names (plant species)
class_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava', 'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica_Cherry-Gasagase', 'Jamun', 'Jasmine', 'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal', 'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']
scientific = ['Amaranthus Viridis','Basella Alba','Piper Betle','Tabernaemontana Divaricata','Murraya Koenigii','Moringa Oleifera','Trigonella Foenum-graecum','Psidium Guajava', 'Hibiscus Rosa-sinensis','Pongamia Pinnata','Brassica Juncea','Artocarpus Heterophyllus', 'Muntingia Calabura','Syzygium Cumini','Jasminum', 'Carissa Carandas','Citrus Limon','Mangifera Indica', 'Plectranthus Amboinicus','Mentha','Azadirachta Indica', 'Nerium Oleander','Nyctanthes Arbor-tristis','Ficus Religiosa', 'Punica Granatum','Alpinia Galanga', 'Syzygium Jambos','Ficus Auriculata', 'Santalum Album ','Ocimum Tenuiflorum']
    
# def predict(img, model):
#     #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
#     #thresh, imgBW = cv2.threshold(imgBlur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     #imgInv = cv2.bitwise_not(imgBW)
#     #kernel = np.ones((50, 50))
#     #imgClosed = cv2.morphologyEx(imgInv, cv2.MORPH_CLOSE, kernel)
#     # Resize
#     img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#     #Adding third dimension to shape
#     #new.shape = (1,) + new.shape + (1, )
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img / 255.0 
#     #print(new.shape, img.shape,flush=True)
#     prediction = model.predict(img)
#     predicted_class_index = np.argmax(prediction)
#     predicted_class = class_names[predicted_class_index]
#     confidence = prediction[0][predicted_class_index]

#     return predicted_class, confidence
def predict(img):
    # Print the input image shape and values for debugging
    

    # Resize the image to match the model's input size
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Print the resized image shape and values for debugging
    img = np.expand_dims(img, axis=0)
    # Normalize the image
    img = img / 255.0
    # Print the normalized image values for debugging
    
    # Make a prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    sci=scientific[predicted_class_index]

    return sci,predicted_class, confidence


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def pred_image():
    file_img = request.files['file']
    img_data = request.files['file'].read()  # Changed variable name to img_data
    img = Image.open(io.BytesIO(img_data))
# Check if the image format is PNG, and if so, convert it to JPG
    if img.format == 'PNG':
        # Convert PNG to RGB mode and save as JPG
        img = img.convert('RGB')
        img_data = BytesIO()
        img.save(img_data, 'JPEG')
        img_data = img_data.getvalue()
    
    # to remove background
    image1 = np.array(img)
    # Create a mask
    mask = np.zeros(image1.shape[:2], np.uint8)

    # Define a rectangle around the leaf (you may need to adjust the coordinates)
    rect = (10, 10, image1.shape[1] - 20, image1.shape[0] - 20)

    # Initialize the background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(image1, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to create a binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Create an all-white background
    white_bg = np.ones_like(image1) * 255

    # Multiply the original image with the mask to get the result
    result = image1 * mask2[:, :, np.newaxis]

    # Set the non-leaf part to white
    result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    
    img2 = result
    # Saving the uploaded image to the UPLOAD_FOLDER
    filename = secure_filename(file_img.filename[:-3] + "jpg")
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    #
    sci_name,species, confidence = predict(img2)
    print(f'Predicted Class: {species}')
    print("Prediction: ", species, flush=True)
    print(f'Confidence: {confidence:.2f}')
    
    if confidence > 0.66:
        # Web Scraping
        # Web Scraping
        # Web Scraping https://tropical.theferns.info/viewtropical.php?id=Carissa+carandas
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        url1 = 'https://pfaf.org/user/Plant.aspx?LatinName={}'.format(sci_name)
        url2 = 'https://tropical.theferns.info/viewtropical.php?id={}'.format(sci_name)
        page = requests.get(url1,headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        # print(soup)
        tables=soup.find_all('table')
        # print(tables[4])
        #composition 
        try:
            composition=list()
            compositiontable=tables[4]
            coldata=(compositiontable.find_all('tr'))
            for row in coldata:
                rowdata=(row.find_all('td'))
                individualrowdata=[data.text.strip() for data in rowdata]
                composition.append(individualrowdata)
            # print(composition)
            dry_weight_index = composition.index(['Figures in grams (g) or miligrams (mg) per 100g of food.', ''])

            # Extract the relevant data
            desired_data = composition[dry_weight_index + 1:]
        
            # Format and print the data
            # for line in desired_data:
            #     print('\n'.join(line))

            formatted_data = "\n".join(["\n".join(line) for line in desired_data])
            # print(formatted_data)
            # Check if the element with class 'boots2' exists
            string_length = len(formatted_data)

            # Calculate the number of characters to keep (1/3 of the length)
            characters_to_keep = string_length // 3

            # Slice the string to keep only the first 1/3 of the characters
            formatted_data = formatted_data[:characters_to_keep]
            formatted_data =formatted_data.replace("Reference: [ ]Notes", "")
        except IndexError:
            formatted_data = comp.get(species, "Details Not Found")
            # if formatted_data == "Details Not Found":
            #     formatted_data == comp[(species)]
        except ValueError or KeyError:
            formatted_data = comp.get(species, "Details Not Found")
            # if formatted_data == "Details Not Found":
            #     formatted_data == comp[(species)]
        # #medicinal props
        # element = soup.find(class_='boots2')
        # if element:
        #     results = element.find('span')
        #     resarr = list(results.descendants)
        #     try:
        #         medprop = resarr[-1]
        #         medprop = re.sub('\[.*?\]', ' ', medprop)
        #     except IndexError:
        #         medprop = "No Medicinal Properties Available"
        # else:
        #     medprop = "No Medicinal Properties Available"


        #medicinal props
        #edible uses if available

        element = soup.find(class_='boots2')
        try:
            if element:
                results = element.find('span')
                resarr = list(results.descendants)
                try:
                    medprop = resarr[-1]
                    medprop = re.sub('\[.*?\]', ' ', medprop)
                except IndexError:
                    medprop = "No Medicinal Properties Available"
        except Exception as e:
            print(f"Error parsing first website: {str(e)}")
        try:
            page2 = requests.get(url2,headers=headers)
            soup2 = BeautifulSoup(page2.content, 'html.parser')
            
            pagetags=soup2.find('div',class_='PageBox')
            
            paragraphs = soup2.find_all('p')  # Assuming the text is within a <p> tag
            for div in soup2.find_all('div', class_='ref'):
                div.decompose()
            text=""
            for paragraph in paragraphs:
                text=text+(paragraph.text)
            
            # print(text)
            # Define a regular expression pattern to capture the desired text
            pattern = r'Edible Uses(.+?)Medicinal(.+?)Other Uses'

            # Use re.search to find the first match
            match = re.search(pattern, text, re.DOTALL)

            # If a match is found, extract the captured group
            if match:
                edibletext = match.group(1)
                edibleuses = edibletext
                edibleuses = re.sub('\[.*?\]', ' ', edibleuses)
                medicinal_text = match.group(2)
                medprop=(medicinal_text)
                medprop = re.sub('\[.*?\]', ' ', medprop)
            # Initialize variables to keep track of h3 tags
            # Print the data
            # print(edibleuses)
            
        except:
            print("Error123 ")
        
        #edible uses if available
        element2 = soup.find(class_='boots3')
        if element2:
            results2 = element2.find('span')
            resarr = list(results2.descendants)
            try:
                edibleuses = resarr[-1]
                edibleuses = re.sub('\[.*?\]', ' ', edibleuses)
            except IndexError:
                print("no uses")
        else:
            print("no uses")  
        # Define filename here before the return statement
        filename = file_img.filename[:-3] + "jpg"
        sci_name = sci_name.replace('+', ' ')

        return render_template("pred.html", medprop=medprop, species=species, sci_name=sci_name, img=filename,formatted_data=formatted_data,edibleuses=edibleuses)
    else:
        return render_template("notfound.html")

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
