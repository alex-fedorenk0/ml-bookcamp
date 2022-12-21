# Project and dataset description

In this project I have used 'is that santa' (https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification) dataset from Kaggle.
The aim is binary image classification.

I have not put the dataset into the repo, so if you want to replicate model training part of the project,
 you can download it either with kaggle cli or manually

# Model selection

As a base model I have used Xception. 
After basic tuning this model has shown a good performance on testing dataset.

# Dependency management

Dependencies for model deployment are listed in [Pipfile](Pipfile) and can be installed via ``` pipenv install ```

# Deployment

The model is deployed with TFServing and Docker Compose

Two docker images for model and gateway are built

```
docker build -t santa-class-model:001 -f image-model.dockerfile .
docker build -t santa-class-gateway:001 -f image-gateway.dockerfile .
```

and run with docker-compose:

```
docker compose up
```

The local deployment can be tested with [test.py](test.py) using any external image in url field.



