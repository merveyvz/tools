import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph, Neo4jVector, GraphCypherQAChain
from uuid import uuid4
from dotenv import load_dotenv
import urllib.parse # URL encode için


load_dotenv()

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4")

embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# schema = graph.schema

schema = """
    Node properties are the following:
    Work {workId: INTEGER (unique), title: STRING, original_language: STRING, description: STRING, embedding: VECTOR (for description similarity)}
    Edition {editionId: INTEGER (unique), isbn: STRING (unique for non-null), title: STRING, publication_date: Date, pages: INTEGER, language: STRING, format: STRING, avg_rating: FLOAT}
    Contributor {contributorId: INTEGER (unique), name: STRING, born: Date, nationality: STRING, died_date: Date, null if alive)}
      - A Contributor can have additional labels: :Author, :Translator, :Illustrator, :Editor
    Award {awardId: INTEGER (unique), name: STRING, category: STRING}
    Publisher {publisherId: INTEGER (unique), name: STRING (unique), country: STRING, founded_year: INTEGER}
    BookSeries {seriesId: INTEGER (unique), name: STRING (unique)}
    Topic {topicId: INTEGER (unique), name: STRING (unique), type: STRING (e.g., 'genre', 'tag')}
    Reader {readerId: INTEGER (unique), name: STRING, age_group: STRING}
    Review {reviewId: INTEGER (unique), rating: FLOAT, content: STRING, review_date: DateTime}

    Relationship properties are the following:
    TRANSLATED_BY {source_language_of_this_translation: STRING}
    ILLUSTRATED_BY {style: LIST<STRING>}
    WON {year: INTEGER}
    PART_OF_SERIES {order: INTEGER}
    READ {status: STRING (e.g., 'has_read', 'currently_reading', 'wants_to_read'), status_update_date: DateTime}
    RELATED_TO {type: LIST<STRING> (e.g., 'inspired_by', 'references', 'same_world_background'), note: STRING}

    The relationships are the following:
    (:Author:Contributor)-[:CREATED]->(:Work)
    (:Edition)-[:IS_EDITION_OF]->(:Work)
    (:Edition)-[:TRANSLATED_BY]->(:Translator:Contributor)
    (:Edition)-[:ILLUSTRATED_BY]->(:Illustrator:Contributor)
    (:Edition)-[:EDITED_BY]->(:Editor:Contributor)
    (:Work)-[:WON]->(:Award)
    (:Edition)-[:PUBLISHED_BY]->(:Publisher)
    (:Edition)-[:PART_OF_SERIES]->(:BookSeries)
    (:Topic)-[:SUBCATEGORY_OF]->(:Topic)
    (:Work)-[:HAS_TOPIC]->(:Topic)
    (:Reader)-[:LIKES_TOPIC]->(:Topic)
    (:Reader)-[:READ]->(:Edition) // READ is on Edition
    (:Reader)-[:WROTE]->(:Review)
    (:Review)-[:REVIEWS]->(:Edition) // REVIEWS is on Edition
    (:Work)-[:RELATED_TO]->(:Work)
    """

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query writer.
Given a question and a database schema, write a Cypher query that can answer the question.
Do not use any node labels, relationship types, or properties that are not explicitly in the schema.
When asked about 'author', 'translator', 'illustrator', or 'editor', you MUST use the corresponding label in combination with the 'Contributor' label. For example, for authors, MATCH (c:Author:Contributor).
If the user asks about a "book" in a general sense (e.g., "books by Tolkien", "books about dragons"), assume they mean a 'Work' node unless specific edition details (like ISBN, publisher, publication date, format, page count, or language of a specific copy) are requested, in which case you should query 'Edition' nodes.
Relationships like READ and REVIEWS are connected to 'Edition' nodes, not 'Work' nodes. If a question implies reading history or specific reviews, you must go through 'Edition'.
Pay attention to the direction of relationships.
The 'embedding' property on 'Work' nodes is for vector similarity. Use `vector.similarity.cosine(work1.embedding, work2.embedding)` if semantic similarity between work descriptions is implied.
If the question cannot be answered with the provided schema, or if the Cypher query would be too complex for a direct answer (e.g., requiring multiple levels of aggregation not easily summarized), output "SCHEMA_LIMITATION" instead of a query.
Output ONLY the Cypher query. Do not add any commentary, preamble, or explanation.

When a question implies multiple conditions on the same node (e.g., a reader has READ an edition AND WROTE a review for THE SAME edition), ensure your MATCH clauses correctly link these conditions.
If a user asks for information that might not exist for all items (e.g., "list books read and their reviews, if any"), use OPTIONAL MATCH for the optional part (e.g., the review).
Only include nodes in the MATCH path if their properties are explicitly requested or are necessary to traverse to requested information. For example, if only an edition's title and its review are asked for, do not necessarily include the Work node unless the work's title is also asked for.


Here are some examples:

Question: "Show me books John Doe read and his reviews for them, if he wrote any."
Cypher Query: MATCH (r:Reader {{ name: "John Doe"}})-[:READ]->(e:Edition)
OPTIONAL MATCH (r)-[:WROTE]->(rv:Review)-[:REVIEWS]->(e)
RETURN e.title AS Book, rv.content AS Review

Question: "Which books by Tolkien also have illustrations by him?"
Cypher Query: MATCH (author:Author:Contributor {{ name: "J.R.R. Tolkien"}})-[:CREATED]->(w:Work)
MATCH (edition:Edition)-[:IS_EDITION_OF]->(w)
MATCH (edition)-[:ILLUSTRATED_BY]->(author) // Re-matching author for the illustrator role on the edition
RETURN DISTINCT w.title AS WorkTitle, edition.title AS EditionTitle


Schema:{schema}

Question: {question}
Cypher Query:"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

# GraphCypherQAChain Oluşturma
cypher_qa_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_generation_prompt,
    allow_dangerous_requests=True, # Tehlikeli sorgulara izin ver
    return_intermediate_steps=True, # Üretilen Cypher'ı ve sonucu görmek için
    return_direct=False # False ise llm ile doğal dilde cevap üretir
)

# Genel Kitap Sohbeti
book_expert_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful book expert and literary assistant."), ("human", "{input}")]
)

book_chat_chain = book_expert_prompt | llm | StrOutputParser()

# Kitap Açıklaması için Vektör İndeksi
work_description_vector_index = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="work_embeddings",
    embedding_node_property="embedding",
    text_node_property="description"
)


# Kitap Açıklaması Benzerlik Arama Aracı
description_retriever_chain = RetrievalQA.from_llm(
    llm=llm,
    retriever=work_description_vector_index.as_retriever(),
    return_source_documents=False
)


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


def generate_book_purchase_links(book_title_or_query: str) -> str:
    print(f"--- Generating purchase/search links for: {book_title_or_query} ---")
    encoded_query = urllib.parse.quote_plus(book_title_or_query)

    links = []

    # Amazon Türkiye
    amazon_tr_url = f"https://www.amazon.com.tr/s?k={encoded_query}&i=stripbooks"
    links.append(f"Amazon Türkiye'de Ara: {amazon_tr_url}")

    # Kitapyurdu
    kitapyurdu_url = f"https://www.kitapyurdu.com/index.php?route=product/search&filter_name={encoded_query}"
    links.append(f"Kitapyurdu'nda Ara: {kitapyurdu_url}")

    if not links:
        return "Üzgünüm, bu kitap için şu anda bir satın alma veya arama linki oluşturamıyorum."

    response_text = f"'{book_title_or_query}' için bazı arama linkleri:\n" + "\n".join(links)
    return response_text

tools = [
    Tool(
        name="Knowledge Graph Query Tool",
        description="""Use this tool ONLY when the user asks a question that requires specific factual information DIRECTLY from the book knowledge graph.
        This includes questions about specific books (Works or Editions), authors, translators, illustrators, editors, series, topics, publishers, awards, reader interactions (likes, reads), or reviews.
        Examples:
        - "What books did J.R.R. Tolkien write?"
        - "Tell me about the editions of 'The Hobbit'."
        - "Which books are in the 'Foundation Series'?"
        - "Who translated the Turkish edition of 'Dune'?"
        - "What are the main topics of '1984'?"
        - "Which publisher released the English version of 'The Silmarillion'?"
        - "Did Elif Yılmaz read 'The Return of the King'?"
        Input should be the user's full question in natural language.
        If the question is a general statement, a greeting, or does not ask for specific data from the graph, DO NOT use this tool.
        """,
        func=lambda q: cypher_qa_chain.invoke({"query": q}).get("result", "I couldn't find a specific answer from the graph for that. You can try rephrasing your question."),
    ),
    Tool(
        name="Book Description Similarity Search",
        description="Use this tool when a user asks for books similar to a given plot, theme, or description, or asks 'What book is this description about?'. The input is the description or plot provided by the user as a string.",
        func=lambda q: description_retriever_chain.invoke({"query": q}),
    ),
    Tool(
        name="Book Purchase Search",
        description="Use when a user asks where they can find, buy, or purchase a specific book. The input should ideally be the title of the book. Returns search links to popular book-related websites.",
        func=generate_book_purchase_links
    ),
    Tool(
        name="General Conversation Tool",
        description="Use this tool for general conversation, greetings, or for questions that DO NOT require fetching specific data from the knowledge graph or other specialized tools. This is the fallback tool.",
        func=book_chat_chain.invoke,
    ),
]


agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=7,
    return_intermediate_steps=True # Agent'ın adımlarını ve araç çıktılarını görmek için
)

chat_agent_with_history  = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

print("Kitap Chatbot'u hazır! Çıkmak için 'exit' yazın.")
while True:
    q = input("> ")
    if q.lower() == "exit":
        break
    try:
        response = chat_agent_with_history.invoke(
            {"input": q},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        print("--- Agent Output ---")
        print(response.get("output", "Bir sorun oluştu, cevap alınamadı."))


    except Exception as e:
        print(f"Bir hata oluştu: {e}")