# Use the official Python image as the base image
FROM python:3-slim

# Set the working directory to /app
WORKDIR /app

# Copy the index.py and requirements.txt files into the container
COPY api/index.py requirements.txt ./

# Install any required packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port the app will run on
EXPOSE 8080

# Define the command to run the app
CMD ["flask", "--app", "index", "run", "--host", "0.0.0.0", "--port", "8080"]
