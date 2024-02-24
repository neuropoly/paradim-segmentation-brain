#  Use an Ubuntu image
FROM freesurfer-wmhsyntheseg

# Install git and other necessary dependencies
RUN apt-get update && \
    apt-get install -y git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Sets the working directory to /app
WORKDIR /app
COPY . /app

# Update pip and setuptools
RUN pip3 install --upgrade pip setuptools

# Install other dependencies
RUN pip3 install -r requirements.txt 
RUN pip3 install -e . 

COPY run.sh /app/run.sh

# Sets the default command to run when the container starts up
ENTRYPOINT ["/app/run.sh"]
