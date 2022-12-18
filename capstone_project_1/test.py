import requests

url = 'http://localhost:9696/predict'

data = {
    'url': 'https://img.freepik.com/free-photo/santa-claus-with-bag-showing-thumb-up_7502-5186.jpg'
}

result = requests.post(url, json=data).json()
print(result)