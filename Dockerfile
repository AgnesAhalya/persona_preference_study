FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml questions.json experiment_utils.py run_experiment.py run_judges.py analyze.py ./
COPY prompts/ ./prompts/
RUN mkdir -p results

CMD ["python", "run_experiment.py", "--dry-run"]
