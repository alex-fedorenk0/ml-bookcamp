import numpy as np

from locust import task, between, HttpUser

sample = {
    "seniority": 10,
    "home": "rent",
    "time": 60,
    "age": 28,
    "marital": "married",
    "records": "no",
    "job": "fixed",
    "expenses": 78,
    "income": 325.0,
    "assets": 18.0,
    "debt": 3000.0,
    "amount": 2250,
    "price": 2250
    }

class CreditRiskTestUser(HttpUser):

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)