import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

#url = 'https://www.helikon-tex.com/media/catalog/product/cache/4/image/9df78eab33525d08d6e5fb8d27136e95/s/p/sp-pgm-dc-11.jpg'

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299, 299))

def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return dict(zip(classes, preds[0]))

def lambda_handler(event, context):
    url = event['url']
    return predict(url)
