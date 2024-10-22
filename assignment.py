# Import necessary libraries
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Step 1: Data Extraction using Selenium and BeautifulSoup
def extract_data_from_url(url):
    # Set up Selenium options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # Path to your ChromeDriver
    driver_path = r'C:\Users\susha\Desktop\triluxo\chromedriver.exe'

    # Initialize the WebDriver
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Fetch the page
    driver.get(url)

    # Use WebDriverWait to ensure elements are loaded before scraping
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "course-card"))
        )
    except:
        print("Timed out waiting for page to load")

    # Extract page source after JavaScript rendering
    page_source = driver.page_source

    # Use BeautifulSoup to parse the page content
    soup = BeautifulSoup(page_source, 'html.parser')

    # Print the HTML to debug the structure
    print(soup.prettify())  # Uncomment to print the HTML structure for inspection

    # Extract relevant data, such as course titles and descriptions
    courses = []
    for course in soup.find_all('div', class_='course-card'):
        title = course.find('h3').get_text(strip=True) if course.find('h3') else "No title"
        description = course.find('p').get_text(strip=True) if course.find('p') else "No description"
        courses.append(f"{title}: {description}")
    
    driver.quit()  # Close the browser
    
    if not courses:
        print("No data found on the page")
    else:
        print(f"Extracted {len(courses)} courses")
    
    return courses


# Step 2: Create embeddings and store them in FAISS vector store
def create_and_store_embeddings(texts):
    if not texts:
        raise ValueError("No texts provided for embedding.")
    
    # Create embeddings using OpenAI model (make sure to set your OpenAI API key)
    embeddings = OpenAIEmbeddings()
    
    # Generate embeddings for each course description
    embedded_texts = embeddings.embed_documents(texts)
    
    if not embedded_texts or len(embedded_texts) == 0:
        raise ValueError("Embeddings generation failed. Check your API key or data.")
    
    # Create a FAISS vector store from the embeddings
    vector_store = FAISS.from_texts(texts, embeddings)
    
    # Save the FAISS index locally
    vector_store.save_local('faiss_index')
    
    return vector_store

from flask import Flask, request, jsonify
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Step 3: Flask REST API for the chatbot
app = Flask(__name__)

# Load FAISS index and embeddings before every request
@app.before_request
def load_faiss_index():
    global vector_store
    if not hasattr(app, 'vector_store_loaded'):
        # Load the FAISS index from local storage if not already loaded
        vector_store = FAISS.load_local('faiss_index', OpenAIEmbeddings())
        app.vector_store_loaded = True  # Mark as loaded to avoid reloading

# Chatbot API route to handle user queries
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    
    # Perform similarity search in the vector store
    docs = vector_store.similarity_search(user_input, k=3)
    
    # Generate responses based on search results
    response = {"responses": [doc.page_content for doc in docs]}
    
    return jsonify(response)

# Main entry point for running the Flask app
if __name__ == '__main__':
    # Extract data from the provided URL
    url = "https://brainlox.com/courses/category/technical"
    course_data = extract_data_from_url(url)
    
    # Check if we have extracted data correctly
    if not course_data:
        raise ValueError("No course data extracted from the URL.")
    
    # Create and store embeddings in the FAISS vector store
    vector_store = create_and_store_embeddings(course_data)
    
    # Run the Flask app
    app.run(debug=True)