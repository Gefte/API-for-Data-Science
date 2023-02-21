FROM python:3.10

# Define a working directory for the application
WORKDIR /app

# Copy the application files to the container
COPY main.py /app/
COPY models /app/models
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
