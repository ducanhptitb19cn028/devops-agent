FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent.py .
COPY ml_analyzer.py .
USER 1000:1000
CMD ["python", "-u", "agent.py"]
