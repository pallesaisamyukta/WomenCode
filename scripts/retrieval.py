import streamlit as st
from pinecone import Pinecone
import json
from sentence_transformers import SentenceTransformer
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class PromptGenerator:
    """
    Class to generate prompts for the user question.
    """

    def __init__(self, json_path):
        """
        Initialize PromptGenerator object.

        Parameters:
        - sentences (list): List of sentences for context.
        - model (str): Path to the Sentence Transformer model.
        - faiss_index: Loaded Faiss index object.
        """
        self.json_path = json_path
        self.sentences = []
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.pipe = pipeline("text2text-generation", model="SaiSamyuktaPalle/RecipeGeneration-BART")
        self.BART_tokenizer = AutoTokenizer.from_pretrained("SaiSamyuktaPalle/RecipeGeneration-BART")
        self.BART_model = AutoModelForSeq2SeqLM.from_pretrained("SaiSamyuktaPalle/RecipeGeneration-BART")
        pc = Pinecone(api_key="35cc5225-2c67-4e0f-b9c9-6f914f64ee22")
        self.pinecone_index = pc.Index("womenhealth")


    def _extract_chunks_content(self):
        """
        Extract chunks from the JSON file.

        Returns:
        - list: List of sentences extracted from the JSON file.
        """

        print("Inside _extract_chunks_content")

        # Read the JSON file and load its content into a dictionary
        with open(self.json_path, 'r') as json_file:
            chunks = json.load(json_file)
        self.sentences = [chunk['text'] for chunk in chunks]

    
    def top_k_sentences(self, query_embedding, k=5):
        """
        Retrieve the top k nearest sentences based on the query embedding.

        Parameters:
        - query_embedding (numpy.ndarray): The embedding of the query.
        - k (int): Number of nearest neighbors to retrieve. Default is 5.

        Returns:
        - list: List of top k nearest sentences.
        """
        # Query Pinecone index for the top k nearest neighbors
        results = self.pinecone_index.query(vector=query_embedding.tolist(), top_k=k)
        # Extract the IDs of the top k similar sentences
        top_k_ids = [match['id'] for match in results['matches']]
        
        # Retrieve the top k nearest sentences using their IDs
        top_k_sentences = [self.sentences[int(idx)] for idx in top_k_ids]
        
        return top_k_sentences


    def rag_prompt(self, question, prev_chat, k=5):
        """
        Generate a prompt for the user question.

        Parameters:
        - question (str): User question.
        - k (int): Number of nearest neighbors to retrieve.

        Returns:
        - str: Prompt containing the question and context.
        """
        print("Inside generate prompt")
        self._extract_chunks_content() #Retrive the chunks and form sentences
        query_embedding = self.model.encode([question + prev_chat])
        context = self.top_k_sentences(query_embedding, k)
        print(context)

        if re.search(r'\b(it |that |this )\b', question, re.IGNORECASE):
            question = question + " " + prev_chat
        # print(context)
        base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using only the useful parts of the provided contexts. 
        Your answers are direct, correct, high-quality, and written by an domain expert for general public. If the provided context does not contain the answer, simply state, 
        "The provided context does not have the answer."

        User question: {}

        Contexts:
        {}
        """
        prompt = base_prompt.format(question, context)
        return prompt
    
    def recipe_details_prompt(self, user_input):
        #Encode the text using the tokenizer
        inputs = self.BART_tokenizer.encode(user_input, return_tensors="pt", max_length=64, truncation=True)
        # Generate outputs
        output_sequences = self.BART_model.generate(
            inputs, 
            max_length=512, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            early_stopping=True
        )

        # Decode the output sequences to get the generated text
        generated_text = self.BART_tokenizer.decode(output_sequences[0], skip_special_tokens=True)        

        base_prompt = """You are an AI assistant. Your task is to format the given recipe. Your answers are direct, correct, high-quality, and written by an domain expert for general public. If the provided context does not contain the answer, simply state, 
        "The provided context does not have the answer."

        Recipe:
        {}
        """
        prompt = base_prompt.format(generated_text)
        return prompt

