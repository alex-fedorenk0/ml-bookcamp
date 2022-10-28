# Midterm project for ML-Zoomcamp 2022

## Dataset and problem description

In this project the Student Performance Dataset from Kaggle is used.
The original work can be found here:
[https://www.researchgate.net/publication/228780408_Using_data_mining_to_predict_secondary_school_student_performance](https://www.researchgate.net/publication/228780408_Using_data_mining_to_predict_secondary_school_student_performance)

The aim is to select and train a model for prediction of student 
performance score based on various numeric and categorical features.

We will try to estimate the impact of different factors on students performance.
The final model can be used, for example, for live tracking of student study and 
giving special attention to students whose results strongly differ from predicted.

There are two separate datasets for math and Portuguese language courses,
with significant part of students (382) belonging in both datasets. As target
variable in two datasets describes performance for two different courses, 
it would be not correct to sipmply merge them. Instead it will be interesting to see
whether model created on one dataset will have perform similarly on data from
another course.

Dataset attributes are described thoroughly in [*student.txt*](student.txt).
Target variable is __G3__ (final grade).

