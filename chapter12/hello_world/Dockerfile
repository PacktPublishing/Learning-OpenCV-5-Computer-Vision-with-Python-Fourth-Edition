FROM public.ecr.aws/lambda/python:3.13

# Install pinned dependencies from requirements.txt. The contrib build is
# required under OpenCV 5 for cv2.CascadeClassifier.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code and Haar cascade classifier.
COPY app.py cascade.xml ${LAMBDA_TASK_ROOT}/

CMD ["app.lambda_handler"]
