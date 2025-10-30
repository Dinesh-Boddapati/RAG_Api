 
# 1. Start from an official, slim Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY main.py .

# 5. Expose the port your app runs on
EXPOSE 8000

# 6. The command to run your application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# This command tells uvicorn to use the $PORT variable provided by Cloud Run
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]