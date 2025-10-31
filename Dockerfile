 

FROM python:3.11-slim

#  Set the working directory inside the container
WORKDIR /app

#  requirements file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of my application code
COPY main.py .

#  Expose the port app runs on
EXPOSE 8000

# The command to run application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# This command tells uvicorn to use the $PORT variable provided by Cloud Run
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
