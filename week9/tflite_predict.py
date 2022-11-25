import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

path = '../week8/clothing-dataset-small/test/hat/2a12baab-f020-42e3-8e6b-5d82e3ed0b55.jpg'

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299, 299))

X = preprocessor.from_path(path)

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

print(dict(zip(classes, preds[0])))


