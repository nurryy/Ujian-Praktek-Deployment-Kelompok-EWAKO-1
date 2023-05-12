'''
	Contoh Deloyment untuk Domain Computer Vision (CV)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================
from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
import os.path
from PIL import Image
from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.applications import MobileNetV2
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from PIL import Image
from fungsi import make_model

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

dataset = './train/fire_dataset/'

image_dir = Path(dataset)

# Get filepaths dan labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png'))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Gabungkan filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename)
	
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = '/static/images/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image         = Image.open('.' + gambar_prediksi)
			
			# Mengubah Ukuran Gambar
			test_image_resized = test_image.resize((32, 32))
			
			# Konversi Gambar ke Array
			image_array        = np.array(test_image_resized)
			test_image_x       = (image_array / 255) - 0.5
			test_image_x       = np.array([image_array])
			
			# Prediksi Gambar
			y_pred_test_single         = model.predict(test_image_x)			
			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
			hasil_prediksi = image_df[y_pred_test_classes_single[0]]
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})

# =[Main]========================================		

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	
	model = make_model()
	model.load_weights("model_fire_dataset.h5")

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)
	
	


