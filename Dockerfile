# Description: Dockerfile for Python 3.12
FROM python:3.12.2

# Set the working directory
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install Rust and required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    # Add OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install vtracer
RUN pip install vtracer

# Create necessary directories with proper permissions
RUN mkdir -p /home/user/app/uploads \
    /home/user/app/outputs \
    /home/user/app/svg_outputs
    

# Create a user
RUN useradd -m user && chown -R user:user /home/user

# Change the ownership of the /code directory to the user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics

# Set the working directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at /home/user/app
COPY --chown=user . $HOME/app

EXPOSE 8080
# Expose the port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
