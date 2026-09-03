FROM public.ecr.aws/lambda/python:3.13

# Install OpenCV (headless – no GUI dependencies needed in Lambda)
# and requests for downloading images from URLs
RUN pip install opencv-python-headless requests

# Copy function code and Haar cascade classifier
COPY app.py cascade.xml ${LAMBDA_TASK_ROOT}/

CMD ["app.lambda_handler"]
