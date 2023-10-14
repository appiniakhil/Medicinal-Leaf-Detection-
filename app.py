from flask import Flask, render_template, request, send_from_directory
import os
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import io
import requests
from bs4 import BeautifulSoup
import re
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

IMAGE_SIZE = 224

# Loading model
model = tf.keras.models.load_model('leaf_classification_model2.h5')  # Replace with the path to your trained model file
model.make_predict_function()

# Define the UPLOAD_FOLDER as a relative path
UPLOAD_FOLDER = 'upload'

# Check if the 'upload' folder exists; if not, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define class names (plant species)
class_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava', 'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica_Cherry-Gasagase', 'Jamun', 'Jasmine', 'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal', 'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']
scientific = ['Amaranthus Viridis', 'Basella Alba', 'Piper Betle', 'Tabernaemontana Divaricata', 'Murraya Koenigii', 'Moringa Oleifera', 'Trigonella Foenum-graecum', 'Psidium Guajava', 'Hibiscus Rosa-sinensis', 'Pongamia Pinnata', 'Brassica Juncea', 'Artocarpus Heterophyllus', 'Muntingia Calabura', 'Syzygium Cumini', 'Jasminum', 'Carissa Carandas', 'Citrus Limon', 'Mangifera Indica', 'Plectranthus Amboinicus', 'Mentha', 'Azadirachta Indica', 'Nerium Oleander', 'Nyctanthes Arbor-tristis', 'Ficus Religiosa', 'Punica Granatum', 'Alpinia Galanga', 'Syzygium Jambos', 'Ficus Auriculata', 'Santalum Album', 'Ocimum Tenuiflorum']

def predict(img):
    # Resize the image to match the model's input size
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make a prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    sci = scientific[predicted_class_index]

    return sci, predicted_class, confidence

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def pred_image():
    file_img = request.files['file']
    img_data = request.files['file'].read()
    img = Image.open(io.BytesIO(img_data))

    # Initialize medprop and edibleuses
    medprop = "no data"
    edibleuses = "no data"

    # Image processing to remove background
    image1 = np.array(img)
    mask = np.zeros(image1.shape[:2], np.uint8)
    rect = (10, 10, image1.shape[1] - 20, image1.shape[0] - 20)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image1, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    white_bg = np.ones_like(image1) * 255
    result = image1 * mask2[:, :, np.newaxis]
    img2 = result

    filename = secure_filename(file_img.filename)
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    sci_name, species, confidence = predict(img2)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    url1 = 'https://pfaf.org/user/Plant.aspx?LatinName={}'.format(sci_name)
    url2 = 'https://tropical.theferns.info/viewtropical.php?id={}'.format(sci_name)

    page = requests.get(url1, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    tables = soup.find_all('table')

    try:
        composition = list()
        compositiontable = tables[4]
        coldata = compositiontable.find_all('tr')
        for row in coldata:
            rowdata = row.findall('td')
            individualrowdata = [data.text.strip() for data in rowdata]
            composition.append(individualrowdata)
        dry_weight_index = composition.index(['Leaves (Dry weight)', ''])
        desired_data = composition[dry_weight_index + 1:]
        formatted_data = "\n".join(["\n".join(line) for line in desired_data])[:len(formatted_data) // 3]
    except IndexError:
        formatted_data = "Details Not Found"
    except ValueError:
        formatted_data = "Details Not Found"

    try:
        page2 = requests.get(url2, headers=headers)
        soup2 = BeautifulSoup(page2.content, 'html.parser')
        
        pagetags = soup2.find('div', class_='PageBox')
        
        paragraphs = soup2.find_all('p')
        for div in soup2.find_all('div', class_='ref'):
            div.decompose()
        text = ""
        for paragraph in paragraphs:
            text = text + (paragraph.text)
        
        pattern = r'Edible Uses(.+?)Medicinal(.+?)Other Uses'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            edibletext = match.group(1)
            edibleuses = edibletext
            edibleuses = re.sub('\[.*?\]', ' ', edibleuses)
            medicinal_text = match.group(2)
            medprop = medicinal_text
            medprop = re.sub('\[.*?\]', ' ', medprop)
    except Exception as e:
        print("Error:", str(e))
    
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
    
    filename = file_img.filename[:-3] + "jpg"
    sci_name = sci_name.replace('+', ' ')

    return render_template("pred.html", medprop=medprop, species=species, sci_name=sci_name, img=filename, formatted_data=formatted_data, edibleuses=edibleuses)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
