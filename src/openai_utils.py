import os
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("azure_open_api_key")
api_version = os.getenv("azure_open_api_version")
azure_endpoint = os.getenv("azure_open_api_endpoint")
embedding_deployment = os.getenv("embedding_deployment")
chat_completition_deployment = os.getenv("chat_completition_deployment")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )
    logger.info("Azure OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    raise


def interpret_topic_with_openai(topic_top_words, topic_top_docs=None):

    system_prompt= """You are a assistant that provides topic name and description given the user input. 
    User input contain: 
     -- top words: the most important words under a topic distribution
     -- documents (if provided): exhibit the topic, however, documents can exhibit other topics as well.
    You only need to describe the topic related to the top words amd use the documents for understanding the relationship of the words.
    Generate a JSON object containing:
    "topic_name": A concise name for the topic derived from the keywords.
    "topic_description": A detailed explanation of the topic, including core ideas, key concepts, and examples highlighted by the documents
    """

    user_prompt= f"""top words:{topic_top_words}"""
    if topic_top_docs != None :
        user_prompt+=f"""example documents: {topic_top_docs},"""
    user_prompt+= """
    Using this information, provide a JSON object with: 
    "topic_name": A concise name for the topic based on the keywords.
    "topic_description": A detailed explanation of the topic, incorporating the meaning of the keywords and relevant insights from the example documents.
    """

    response=client.chat.completions.create(model=chat_completition_deployment, 
                            messages=[{"role":"system", "content": system_prompt},
                                        {"role":"user","content":user_prompt}], 
                            temperature=0.0,
                            logprobs=True)
    
    return json.loads(response.choices[0].message.content)

def generate_embeddings(text, model=embedding_deployment):
    """
    Generate embeddings for the given text using the specified model.
    
    Args:
        text (str): The input text to generate embeddings for.
        model (str): The model deployment name to use for generating embeddings.
    
    Returns:
        list: The generated embeddings.
    """
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        logger.info("Embeddings generated successfully.")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

