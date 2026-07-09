FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8510

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8510", "--server.address=0.0.0.0", "--server.headless=true"]
