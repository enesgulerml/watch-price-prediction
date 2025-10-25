# Stage 1: Base Environment
# Use a lightweight Python base image suitable for production
FROM python:3.10-slim-buster 

# 2. METADATA
# Set the working directory for the application code inside the container
WORKDIR /app

# 3. INSTALL DEPENDENCIES
# Copy requirements.txt from the root directory (Build Context)
COPY requirements.txt .
# Install all necessary Python libraries and clear the cache
RUN pip install --no-cache-dir -r requirements.txt

# 4. ENVIRONMENT VARIABLE FOR CONDITIONAL LOGIC
# Set an environment variable to tell api.py that it is running inside Docker.
# This activates the simplified model path in the Python code.
ENV RUNNING_IN_DOCKER=True

# 5. COPY CODE AND RESOURCES (Paths are relative to the Project Root)

# a) MODEL FOLDER COPY:
# Copy the 'models' folder from the root to the Container's root directory (/) 
# to satisfy the model path requirement of "/models/..."
COPY models /models 

# b) APPLICATION CODE COPY:
# Copy Flask API, templates, and static files from the 'flask-app' subdirectory
COPY flask-app/api.py .
COPY flask-app/templates ./templates 
COPY flask-app/static ./static 

# 6. PORTS AND ENTRYPOINT
# Inform Docker that the application listens on port 5000
EXPOSE 5000

# Command to run the API when the container starts
# The API file is now located directly in /app
CMD ["python", "api.py"]