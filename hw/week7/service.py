import bentoml
from bentoml.io import JSON, NumpyNdarray

model_ref = bentoml.sklearn.get('mlzoomcamp_homework:qtzdz3slg6mwwdu5')
model_runner = model_ref.to_runner()
svc = bentoml.Service('mlzoomcamp_homework', runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(input_data):
    pred = await model_runner.predict.async_run(input_data)
    print(pred)

    return {'prediction': pred}