import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

#from keras_image_helper import create_preprocessor

interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


#preprocessor = create_preprocessor('xception', target_size=(150, 150))

def predict(url):
    #X = preprocessor.from_url(url)
    img = download_image(url)
    img = prepare_image(img, (150, 150))
    X = np.array([np.array(img)/255], dtype='float32')

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return preds[0].tolist()

def lambda_handler(event, context):
    url = event['url']
    return predict(url)