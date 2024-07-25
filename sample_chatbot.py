import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np


# 1. Fetch and Process Knowledge Base
def fetch_documents_from_urls(url_list):
    texts = []
    for url in url_list:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        combined_text = ' '.join([p.get_text() for p in paragraphs])
        texts.append(combined_text)
    return texts


# 2. Create Embeddings for Documents
def create_document_embeddings(texts):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    document_embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    return document_embeddings, embedding_model


# 3. Initialize NLP Components
def setup_nlp_pipeline():
    text_generator = pipeline('text-generation', model='gpt-3')
    return text_generator


def get_most_relevant_document_index(query, document_embeddings, embedding_model):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    best_match_index = np.argmax(similarity_scores)
    return best_match_index



def generate_answer(query, texts, document_embeddings, embedding_model, text_generator):
    best_doc_index = get_most_relevant_document_index(query, document_embeddings, embedding_model)
    relevant_text = texts[best_doc_index]
    response = text_generator(f"Based on the following document: {relevant_text} Answer this question: {query}", max_length=150)
    return response[0]['generated_text']


# 4. Main Execution Function

def main():
    urls = [
    "https://docs.motadata.com/motadata-aiops-docs/",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Adding%20Cloud%20Devices%20for%20Monitoring",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Adding%20Network%20Devices%20for%20Monitoring",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Adding%20Servers%20for%20Monitoring",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Adding%20Virtualization%20Devices%20for%20Monitoring",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Adding%20Wireless%20Devices%20for%20Monitoring",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Adding-service-checks-for-monitoring",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Azure-Integration-steps",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Credential%20Profile",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/Discovery%20Profile",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/office-integration-steps",
    "https://docs.motadata.com/motadata-aiops-docs/Adding%20and%20Managing%20Devices/overview",
    "https://docs.motadata.com/motadata-aiops-docs/agent-based-monitoring-system/agent-overview",
    "https://docs.motadata.com/motadata-aiops-docs/agent-based-monitoring-system/architecture-of-agent-based-setup",
    "https://docs.motadata.com/motadata-aiops-docs/agent-based-monitoring-system/metric-log-configuration",
    "https://docs.motadata.com/motadata-aiops-docs/alerts-and-policies/aiops-policies/anomaly-policy",
    "https://docs.motadata.com/motadata-aiops-docs/alerts-and-policies/aiops-policies/forecast-policy",
    "https://docs.motadata.com/motadata-aiops-docs/alerts-and-policies/aiops-policies/overview",
    "https://docs.motadata.com/motadata-aiops-docs/alerts-and-policies/alert-correlation",
    "https://docs.motadata.com/motadata-aiops-docs/alerts-and-policies/basic-policies/availability-policy"
]

    
    
    # Fetch and process the documents from the URLs
    texts = fetch_documents_from_urls(url_list)
    
    # Create embeddings for the fetched documents
    document_embeddings, embedding_model = create_document_embeddings(texts)
    
    # Set up the NLP pipeline for text generation
    text_generator = setup_nlp_pipeline()
    
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break
        
        # Generate and display the response
        answer = generate_answer(user_input, texts, document_embeddings, embedding_model, text_generator)
        print(f"Chatbot: {answer}")

if __name__ == "__main__":
    main()
