# DisasterMgm

**Architecture Diagram**

![image](https://github.com/user-attachments/assets/8c9d1cd4-4aa0-40d0-b7cf-d4474082b0a9)

forecasting.py - Uses Data from Snowflake for Forecasting.

visualize.py - visualizes data based on snowflake data.

main.py - backend code

frontend.py - Frontend code

**Contributions**

Kaushik- Built the snowflake agent and the forecasting agent, where we have taken the disaster data from different sources and stored in snowflake and using the data for various purposes, also did the architecture diagram and documentation.

Arvind- Built the RAG agent which takes PDFs as the input data and converts them into indexings and embeddings and stores them in Pinecone VectorDB, where the data can be accessed by providing prompts, also did documentation and deployment using docker.

Riya - Built the Parse and Data retrieve agent, which retrieves the data and parses them to provide the data as input to the LLMs, also did the Web agent which provides the current and latest disaster data, along with the visualization agent and statistical agent which does mathematical calculations on the data to provide charts and finally a report synthesis agent which checks for all the reports the other agents have done based on specific keywords and generates a full report, also integrated this with Streamlit and FastAPI.

