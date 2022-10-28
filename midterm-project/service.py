import bentoml
from bentoml.io import JSON

model_ref = bentoml.sklearn.get("student_performance_model:5jw4gtswzskqo5cm")
dv = model_ref.custom_objects['dict_vectorizer']
scaler = model_ref.custom_objects['standard_scaler']

model_runner = model_ref.to_runner()

svc = bentoml.Service("student_performance_model", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
async def predict(app_data):
    vector = dv.transform(app_data)
    vector = scaler.transform(vector)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    
    result = prediction[0].round()
    if result > 20:
        result = 20
    if result < 0:
        result = 0

    if result >= 10:
        return {
            "predicted_grade": result,
            "status": "PASS"
        }
    else:
        return {
            "predicted_grade": result,
            "status": "FAIL"
        }