# Use official Python slim image
FROM python:3.12-slim

WORKDIR /app

# Copy required files
COPY app.py app.py
COPY templates/ templates/
COPY static/ static/
COPY house_price_model.pkl house_price_model.pkl
COPY label_encoders.pkl label_encoders.pkl
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
