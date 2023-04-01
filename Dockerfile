# Use the official Python image as the base image
FROM python:3.11-alpine

# Set the working directory to /app
WORKDIR /app

# Copy the main.py and requirements.txt files into the container
COPY main.py requirements.txt ./

# Install any required packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port the app will run on
EXPOSE 8080

# Define the command to run the app
CMD ["functions-framework", "--target", "responder", "--port", "8080"]
