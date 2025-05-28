import os
import csv

from openai import OpenAI
from neo4j import GraphDatabase

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_work_descriptions(limit=None):
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

    driver.verify_connectivity()

    query = """MATCH (w:Work) WHERE w.description IS NOT NULL
               RETURN w.workId AS workId, w.title AS title, 
               w.original_language AS original_language,
               w.description AS description"""

    if limit is not None:
        query += f' LIMIT {limit}'

    works, summary, keys = driver.execute_query(
        query
    )

    driver.close()

    return works

def generate_embeddings(file_name, limit=None):
    csvfile_out = open(file_name, 'w', encoding='utf8', newline='')
    fieldnames = ['workId', 'embedding']
    output_plot = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    output_plot.writeheader()

    works = get_work_descriptions(limit=limit)

    print(len(works))

    llm = OpenAI()

    for work in works:
        print(work['title'])

        description = f"{work['title']}: {work['description']}"
        response = llm.embeddings.create(
            input=description,
            model='text-embedding-ada-002'
        )

        output_plot.writerow({
            'workId': work['workId'],
            'embedding': response.data[0].embedding
        })

    csvfile_out.close()

generate_embeddings('work-desc-embeddings.csv')
