FROM python:3.12.4

WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install sentence-transformers
RUN pip install --no-cache-dir sentence-transformers

RUN pip install --no-cache-dir \
    pandas \
    snowflake-connector-python \
    matplotlib \
    langchain-openai \
    openai

# Copy app code and credentials
COPY . .
COPY starry-tracker-449020-f2-084e3e50b41c.json /app/starry-tracker-449020-f2-084e3e50b41c.json

# Handle pinecone cleanup and reinstallation
RUN pip uninstall -y pinecone pinecone-client pinecone-plugin-inference || true
RUN pip install --no-cache-dir "pinecone-client>=3.0.0,<4.0.0"
# Set Google credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/starry-tracker-449020-f2-084e3e50b41c.json"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
