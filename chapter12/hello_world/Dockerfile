ARG FUNCTION_DIR="/function"

FROM joeminichino/opencv4.6-python3.9 as build-image

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    make \
    cmake \
    unzip \
    libcurl4-openssl-dev

ARG FUNCTION_DIR

# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Copy function code
COPY app.py requirements.txt entry.sh cascade.xml ${FUNCTION_DIR}/

RUN pip install \
    --target ${FUNCTION_DIR} \
    awslambdaric \
    requests


# Multi-stage build: grab a fresh copy of the base image
FROM joeminichino/opencv4.6-python3.9

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the build image dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]

CMD [ "app.lambda_handler" ]