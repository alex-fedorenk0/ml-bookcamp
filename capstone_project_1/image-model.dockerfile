FROM tensorflow/serving:2.7.0

COPY santa-class-v1 /models/santa-class-v1/1

ENV MODEL_NAME="santa-class-v1"