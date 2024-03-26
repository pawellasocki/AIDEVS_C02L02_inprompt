import requests
import json
from icecream import ic
from dotenv import load_dotenv
import os
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tiktoken

load_dotenv('my.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIDEVS_API_KEY = os.getenv("AIDEVS_API_KEY")
url = 'https://tasks.aidevs.pl/'


def authorize(taskname, my_API_KEY):
    return requests.post(url + 'token/' + taskname, data='{"apikey":"' + my_API_KEY + '"}').json()


def get_task_data(token):
    return requests.get(url + 'task/' + token).json()


def send_answer(token, answer):
    return requests.post(url + 'answer/' + token, data=answer)


if __name__ == "__main__":
    '''
    Skorzystaj z API tasks.aidevs.pl, aby pobrać dane zadania inprompt. Znajdziesz w niej dwie właściwości — input, 
    czyli tablicę / listę zdań na temat różnych osób (każde z nich zawiera imię jakiejś osoby) oraz question 
    będące pytaniem na temat jednej z tych osób. Lista jest zbyt duża, aby móc ją wykorzystać w jednym zapytaniu, 
    więc dowolną techniką odfiltruj te zdania, które zawierają wzmiankę na temat osoby wspomnianej w pytaniu. 
    Ostatnim krokiem jest wykorzystanie odfiltrowanych danych jako kontekst na podstawie którego model ma udzielić 
    odpowiedzi na pytanie. Zatem: pobierz listę zdań oraz pytanie, skorzystaj z LLM, aby odnaleźć w pytaniu imię, 
    programistycznie lub z pomocą no-code odfiltruj zdania zawierające to imię. 
    Ostatecznie spraw by model odpowiedział na pytanie, a jego odpowiedź prześlij do naszego API w obiekcie JSON 
    zawierającym jedną właściwość “answer”.
    '''

    taskname = 'inprompt'
    token = authorize(taskname, AIDEVS_API_KEY)['token']
    task_data = get_task_data(token)
    sentences = task_data['input']
    msg = task_data['msg']
    question = task_data['question']

    # RAG parameters
    collection_name = taskname
    embedding_model = 'text-embedding-3-small'
    embedding_size = 1536
    stop_retrieve_token_limit = 500

    openai.api_key = OPENAI_API_KEY


    def get_embedding(sentence):
        response = openai.embeddings.create(
            input=sentence,
            model=embedding_model
        )
        return response.data[0].embedding

    # qdrant client
    client = QdrantClient("localhost", port=6333)
    if collection_name not in [x.name for x in client.get_collections().collections]:
        print(f'Creating collection {taskname}...')
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE),
        )

    # create embeddings @openai and add to qdrant db
    for sentence in sentences:
        sentence_md5 = hashlib.md5(sentence.encode()).hexdigest()
        points = client.retrieve(
            collection_name=collection_name,
            ids=[sentence_md5],
        )
        if not points:
            print(f'Creating point {sentence_md5}...')
            client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=sentence_md5,
                        payload={
                            "source": "AI_DEVS API",
                            "sentence": sentence,
                        },
                        vector=get_embedding(sentence),
                    ),
                ],
            )

    # get question embedding
    question_embedding = get_embedding(question)
    print(f'Question: {question}')

    # search for best matched sentences and build retrieved data
    tiktoken.get_encoding("cl100k_base")  # download encoding from internet
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    retrived_data_size = 0
    retrived_data_content = ''
    offset = 0
    while retrived_data_size < stop_retrieve_token_limit:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            limit=1,
            offset=offset,
        )
        offset += 1
        document = search_result[0].payload['sentence']
        retrived_data_content += document + '\n*\n'
        retrived_data_size += len(encoding.encode(document)) + 3

    system_prompt = msg + '\n###\n' + retrived_data_content
    print('System prompt: ', system_prompt)

    # generate answer
    llm = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", question)])
    chain = prompt | llm
    response = chain.invoke({}).content
    print(f'LLM response: {response}')

    # send answer to API
    answer = send_answer(token, json.dumps({'answer': response}))
    print(f'answer data: {answer.json()}')
    print(f'status code: {answer.status_code}')
