# Use Python 3.12 specifically
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

# Install Rust and required build tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env

# Copy the index.py and requirements.txt files into the container
COPY api/index.py requirements.txt ./

# Install any required packages specified in requirements.txt
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port the app will run on
EXPOSE 8080

# Define the command to run the app
CMD ["flask", "--app", "index", "run", "--host", "0.0.0.0", "--port", "8080"]
