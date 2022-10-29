# Midterm project for ML-Zoomcamp 2022

## Dataset and problem description

In this project the Student Performance Dataset from Kaggle is used.
The original work can be found here:
[https://www.researchgate.net/publication/228780408_Using_data_mining_to_predict_secondary_school_student_performance](https://www.researchgate.net/publication/228780408_Using_data_mining_to_predict_secondary_school_student_performance)

The aim is to select and train a model for prediction of student 
performance score based on various numeric and categorical features.

In process of data analysis and model training I tried to estimate the impact
 of different factors on students performance.
The final model can be used, for example, for live tracking of student study and 
giving special attention to students whose results strongly differ from predicted.

There are two separate datasets for math and Portuguese language courses,
with significant part of students (382) belonging in both datasets. As target
variable in two datasets describes performance for two different courses, 
it would be not correct to simply merge them. 

Dataset attributes are described thoroughly in [*student.txt*](student.txt).
Target variable is __G3__ (final grade).

## Model selection

I have tried several regresion models from sklearn, namely LinearRegression, 
Ridge, SVR and RandomForestRegressor.
The lowest error score on test set was from last, tree-based model.

## Model deployment

Final model was deployed with BentoML, Docker and Heroku
[https://limitless-tundra-85111.herokuapp.com/](https://limitless-tundra-85111.herokuapp.com/)

The code for testing deployed model is in the last section of [notebook.ipynb] (notebook.ipynb),
or it can be tested via the web interface using the next sample data:

``{"school": "GP", "sex": "F", "age": 19, "address": "U", "famsize": "LE3", "Pstatus": "A", "Medu": 2, "Fedu": 3, "Mjob": "at_home", "Fjob": "other", "reason": "home", "guardian": "other", "traveltime": 2, "studytime": 1, "failures": 1, "schoolsup": "no", "famsup": "no", "paid": "no", "activities": "no", "nursery": "yes", "higher": "no", "internet": "yes", "romantic": "no", "famrel": 2, "freetime": 2, "goout": 3, "Dalc": 3, "Walc": 4, "health": 5, "absences": 16}``

