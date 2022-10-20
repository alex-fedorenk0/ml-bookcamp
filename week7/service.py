import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class CreditApp(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: float
    price: float

model_ref = bentoml.xgboost.get('credit_risk_model:z65gwykpu6p2s5cm')

dv = model_ref.custom_objects['dict_vectorizer']
model_runner = model_ref.to_runner()
svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])

@svc.api(input=JSON(pydantic_model=CreditApp), output=JSON())
async def classify(credit_app):
    app_data = credit_app.dict()
    vector = dv.transform(app_data)
    pred = await model_runner.predict.async_run(vector)
    print(pred)

    if pred[0] > 0.5:
        return {'status': 'DECLINED'}
    elif pred[0] > 0.25:
        return {'status': 'MAYBE'}
    else:
        return {'status': 'APPROVED' }