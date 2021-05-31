from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import os
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('inceptionV3.h5')

directory = "archive/images/images"
img_dog = os.listdir(directory)

dog_names = []
for name in img_dog:
    name  = name.split('-')
    dog_names.append(name[1])

numbers = np.arange(0, 120)
num =  zip(numbers, dog_names)
dog_names = dict(num)

@app.route('/')
def home():
	print("home method executed")
	return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		file = request.files['file']
		print("file")
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'img_uploads', secure_filename(file.filename))
		file.save(file_path)

		image = load_img(file_path, target_size=(128, 128))
		image_arr = img_to_array(image)
		image_arr = image_arr/255
		x = np.expand_dims(image_arr, axis=0)
		pred = model.predict(x)
		pred = np.argmax(pred, axis=1)
		print("predicted values is", pred)
		pred = dog_names[pred.tolist()[0]]
	return pred

if __name__ == '__main__':
    app.run(debug=True)