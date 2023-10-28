from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tensorflow as tf
import logging
import sys
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

# Use the built-in Flask static folder for serving static files (CSS, JavaScript, etc.)
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




# Define class names (plant species)
class_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava', 'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica_Cherry-Gasagase', 'Jamun', 'Jasmine', 'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal', 'Pomegranate', 'Rasna', 'Rose_Apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']
scientific = ['Amaranthus Viridis','Basella Alba','Piper Betle','Tabernaemontana Divaricata','Murraya Koenigii','Moringa Oleifera','Trigonella Foenum-graecum','Psidium Guajava', 'Hibiscus Rosa-sinensis','Pongamia Pinnata','Brassica Juncea','Artocarpus Heterophyllus', 'Muntingia Calabura','Syzygium Cumini','Jasminum', 'Carissa Carandas','Citrus Limon','Mangifera Indica', 'Plectranthus Amboinicus','Mentha','Azadirachta Indica', 'Nerium Oleander','Nyctanthes Arbor-tristis','Ficus Religiosa', 'Punica Granatum','Alpinia Galanga', 'Syzygium Jambos','Ficus Auriculata', 'Santalum Album ','Ocimum Tenuiflorum']
comp = {
    # 'Arive-Dantu' : 'Calories: 22 kcal Water: 87.2% Protein: 2.97 g Fat: 0.72 g Carbohydrate: 8.75 g Fiber: 2.5 g Minerals: Potassium, calcium, magnesium, iron, zinc, copper, manganese Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K, folate, niacin, pantothenic acid, pyridoxine, riboflavin, thiamine.',
    #'Basale' : 'Calories: 23 kcal per 100 g Water: 92% Protein: 2 g Fat: 0.2 g Carbohydrate: 5 g Fiber: 2 g Minerals: Potassium, calcium, magnesium, iron, zinc, copper, manganese, phosphorus Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K, vitamin B9 (folic acid), riboflavin, niacin, thiamine.',
    'Betel' : 'Calories: 44 calories Water: 80-85% water Protein: 2.6 grams Fat: 0.9 grams Carbohydrate: 10.4 grams Fiber: 1.9 grams Minerals per 100 grams: (Calcium: 100 mg, Iron: 1.2 mg, Magnesium: 33 mg, Phosphorus: 40 mg, Potassium: 425 mg, Sodium: 10 mg, Zinc: 0.4 mg) Vitamins: (Vitamin A: 100 international units, Vitamin C: 15 mg, Vitamin K: 100 micrograms).',
    'Crape_Jasmine' : 'Calories per 100g: 30-40 Water percentage: 70-80% Protein: 2-3% Fat: 1-2% Carbohydrates: 10-15% Fiber: 2-3% Minerals: Calcium, iron, magnesium, potassium, phosphorus, zinc, and others. Vitamins: A, B, C, and E. Alkaloids: Ibogaine, catharanthine, coronaridine, voacangine, and others. Flavonoids: Quercetin, rutin, and others.',
    'Curry' : 'Calories: 32 kcal per 100 g Water: 63% Protein: 2.5 g Fat: 1 g Carbohydrate: 7.5 g Fiber: 2.5 g Minerals: Calcium, iron, magnesium, phosphorus, potassium, zinc Vitamins: Vitamin A, vitamin B, vitamin C, vitamin E.',
    'Drumstick' : 'Calories: 25 kcal per 100 g  Water: 75%  Protein: 2 g  Fat: 0.5 g    Carbohydrate: 7 g    Fiber: 1 g   Minerals: Potassium, calcium, magnesium, iron, zinc, copper   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    #'Fenugreek' : 'Calories: 22  Water: 84%  Protein: 2.5g  Fat: 0.3g  Carbohydrates: 3.5g  Fiber: 2.2g   Minerals:(Calcium: 60mg,Iron: 3mg,Magnesium: 20mg,Phosphorus: 40mg,Potassium: 300mg)   Vitamin A: 1000IU,Vitamin C: 20mg,Vitamin B1: 0.1mg,Vitamin B2: 0.1mg,Vitamin B3: 0.5mg,Vitamin B9 (folic acid): 60mcg.',
    'Guava' : 'Calories: 49  Water: 82.47%  Protein: 18.53%  Fat: 0.62%  Carbohydrates: 12.74%  Fiber: 2.74%  Minerals:(Calcium: 120 mg,Iron: 2.8 mg,Magnesium: 23 mg,Phosphorus: 34 mg,Potassium: 171 mg,Sodium: 2 mg) Vitamins:(Vitamin A: 100 IU,Vitamin C: 103 mg,Vitamin E: 0.5 mg,Vitamin K: 107 mcg) Folate: 18 mcg  Niacin: 0.7 mg  Riboflavin: 0.1 mg  Thiamine: 0.1 mg.',
    #'Hibiscus' : 'Calories: 30 kcal per 100 g  Water: 70%  Protein: 2.5 g  Fat: 0.5 g Carbohydrate: 7 g  Fiber: 1.5 g  Minerals: Potassium, calcium, magnesium, iron, zinc, copper  Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Indian_Beech' : 'Calories: 25 kcal per 100 g  Water: 70%  Protein: 3.5%  Fat: 1%  Carbohydrates: 25%  Fiber: 3%  Minerals: Calcium, iron, potassium, magnesium, phosphorus, and zinc  Vitamins: Vitamin A, vitamin C, and vitamin B6.',
    'Indian_Mustard' : 'Calories: 25 kcal per 100 g  Water: 85%   Protein: 3 g    Fat: 0.5 g  Carbohydrate: 6 g  Fiber: 1.5 g  Minerals: Calcium, iron, magnesium, potassium, zinc, copper, manganese  Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K, folate.',
    'Jackfruit' : 'Calories: 30 kcal per 100 g   Water: 70%  Protein: 3 g  Fat: 0.5 g   Carbohydrate: 6 g  Fiber: 2 g  Minerals: Potassium, calcium, magnesium, iron, zinc, copper  Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Jamaica_Cherry-Gasagase' : 'Calories: 30 kcal per 100 g   Water: 70%   Protein: 3 g   Fat: 0.5 g   Carbohydrate: 6 g   Fiber: 1.5 g   Minerals: Potassium, calcium, magnesium, iron, zinc, copper, manganese    Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K, vitamin B6, folate.',
    'Jamun': 'Calories: 46   Water: 72%  Protein: 3.1 grams   Fat: 1 gram   Carbohydrate: 20.8 grams   Fiber: 3.7 gramsmg  Minerals:(Calcium: 240,,Iron: 1.2 mg,Magnesium: 28 mg,Phosphorus: 50 mg,Potassium: 370 mg,Sodium: 10 mg,Zinc: 0.5 mg)   Vitamins:(Vitamin A: 200 IU,Vitamin C: 15 mg,Thiamine: 0.1 mg,Riboflavin: 0.2 mg,Niacin: 0.6 mg,Vitamin B6: 0.1 mg,Folate: 10 micrograms,Vitamin K: 10 micrograms).',
    'Jasmine' : 'Calories: 40 kcal per 100 g   Water: 70%   Protein: 2.5 g   Fat: 0.5 g   Carbohydrate: 7 g   Fiber: 1.5 g   Minerals: Potassium, calcium, magnesium, iron, zinc, copper   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Karanda' : 'Calories: 30 kcal per 100 g   Water: 70%   Protein: 2.5 g   Fat: 0.7 g   Carbohydrate: 6.8 g  Fiber: 2 g   Minerals: Potassium, calcium, magnesium, iron, zinc, copper   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
#     'Lemon' : 'Calories: 20 kcal per 100 g   Water: 78%   Protein: 1 g   Fat: 0.2 g   Carbohydrate: 4 g   Fiber: 1 g    Minerals: Potassium, calcium, magnesium, iron, zinc   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Mango'  :  'Calories: 23 kcal per 100 g  Water: 77%  Protein: 2 g   Fat: 0.5 g   Carbohydrate: 5 g   Fiber: 1 g   Minerals: Potassium, calcium, magnesium, iron, zinc, copper  Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Mexican_Mint' : 'Calories: 40   Water percentage: 75%    Protein: 1.5 grams   Fat: 0.5 grams   Carbohydrates: 7 grams   Fiber: 1 gram   Minerals: Calcium, iron, magnesium, phosphorus, and potassium  Vitamins: A, C, and K.',
#     'Mint' : 'Calories: 6 kcal per 100 g  Water: 70%   Protein: 0.4 g   Fat: 0.1 g   Carbohydrate: 1.4 g   Fiber: 0.8 g   Minerals: Potassium, calcium, magnesium, iron, manganese, copper  Vitamins: Vitamin A, vitamin C, vitamin B6, vitamin K.',
    'Neem' : 'Calories: 50 kcal per 100 g  Water: 70%  Protein: 3 g  Fat: 1 g   Carbohydrate: 10 g   Fiber: 2 g   Minerals: Calcium, magnesium, iron, zinc, copper, phosphorus, manganese   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K, riboflavin, niacin, pantothenic acid, pyridoxine.',
    'Oleander' : 'Oleander leaf contains a variety of cardiac glycosides, including oleandrin, neriin, and digitoxigenin. These compounds have a similar effect on the heart as digitalis, a prescription medication used to treat heart failure.Nutritional composition of oleander leaf:  Calories   Water  Protein  Fat  Carbohydrate  Fiber  Minerals  Vitamins.',
    'Parijata' : 'Calories: 20 kcal per 100 g   Water: 75%  Protein: 1.5 g   Fat: 0.3 g  Carbohydrate: 5 g  Fiber: 1 g  Minerals: Potassium, calcium, magnesium, iron, zinc, copper   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Peepal' : 'Calories: 22 kcal per 100 g   Water: 78%  Protein: 2 g   Fat: 0.5 g  Carbohydrate: 6 g   Fiber: 1 g   Minerals: Potassium, calcium, magnesium, iron, zinc, copper   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
#     'Pomegranate' : 'Calories: 35  Water: 70.3 g  Protein: 2.4 g  Fat: 0.4 g   Carbohydrates: 8.3 g   Fiber: 3.5 g   Minerals:(Calcium: 110 mg,Iron: 2.5 mg,Magnesium: 25 mg,Phosphorus: 28 mg,Potassium: 240 mg,Sodium: 11 mg,Zinc: 0.4 mg)   Vitamins:(Vitamin A: 432 IU,Vitamin C: 13 mg,Vitamin K: 150 mcg,Thiamin: 0.07 mg,Riboflavin: 0.05 mg,Niacin: 0.4 mg,Vitamin B6: 0.09 mg,Folate: 18 mcg).',
    'Rasna' : 'Calories: 25 kcal per 100 g   Water: 75%  Protein: 2 g   Fat: 0.5 g   Carbohydrate: 7 g   Fiber: 1 g  Minerals: Potassium, calcium, magnesium, iron, zinc, copper  Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Rose_Apple' : 'The medicinal composition of rose apple leaf is not widely studied, but it is known to contain a variety of bioactive compounds, including:  Polyphenols such as flavonoids, tannins, and stilbenes Terpenoids  Essential oils  Phenolic acids  Vitamin C  Minerals such as potassium, calcium, magnesium, and iron   The exact nutritional composition of rose apple leaf varies depending on the growing conditions, but a typical 100g serving may contain:(Calories: 30-40,Water: 70-80%,Protein: 2-3%,Fat: 0.5-1%,Carbohydrates: 10-15%,Fiber: 2-3%,Minerals: 1-2%,Vitamins: 1-2%).',
    'Roxburgh_fig' : 'Calories: 40 kcal per 100 g  Water: 70%  Protein: 3 g  Fat: 0.5 g   Carbohydrate: 10 g  Fiber: 2 g   Minerals: Potassium, calcium, magnesium, iron, zinc, copper  Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.',
    'Sandalwood'  : 'Calories: 40   Water: 70%  Protein: 2.5 grams  Fat: 1 gram   Carbohydrates: 15 grams   Fiber: 5 grams  Minerals: Calcium, iron, magnesium, potassium, sodium, zinc   Vitamins: Vitamins A, C, and E.',
    'Tulsi' : 'Calories: 66 kcal per 100 g   Water: 64%   Protein: 2.3 g   Fat: 1.1 g   Carbohydrate: 12.7 g   Fiber: 3.3 g   Minerals: Potassium, calcium, magnesium, iron, zinc, manganese, copper   Vitamins: Vitamin A, vitamin C, vitamin E, vitamin K.'
}
    

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
    filename = secure_filename(file_img.filename)
    #cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img2)
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    #
    sci_name,species, confidence = predict(img2)
    print(f'Predicted Class: {species}')
    print("Prediction: ", species, flush=True)
    print(f'Confidence: {confidence:.2f}')
    
    
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
        formatted_data = comp[(species)]
        # if formatted_data == "Details Not Found":
        #     formatted_data == comp[(species)]
    except ValueError or KeyError:
        formatted_data = comp[(species)]
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


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
