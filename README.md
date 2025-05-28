# tools

[`book-chatbot.py`](https://github.com/merveyvz/tools/blob/main/book-chatbot.py) it's a chatbot app, it offers book recommendations with neo4j graph database and agents.

[`work_desc_openai_embeddings.py`](https://github.com/merveyvz/tools/blob/main/work_desc_openai_embeddings.py) converts the data stored in the neo4j graph database into vector embeddings. The query should be updated according to the graph structure.

To run this script, add an `.env` file to the project folder and add these properties
```
OPENAI_API_KEY=
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
```


