import bentoml
from bentoml.io import JSON

model_ref = bentoml.sklearn.get("student_performance_model:p6cpyzkwtg4s25cm")
dv = model_ref.custom_objects['dict_vectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("student_performance_model", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
async def predict(application_data):
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    
    result = prediction[0].round()
    if result > 20:
        result = 20

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