from langchain_community.tools import DuckDuckGoSearchResults
import requests
from bs4 import BeautifulSoup
import re

def get_results(query):
    try:
        search = DuckDuckGoSearchResults(output_format='list')
        list_results = search.invoke(query)
        return list_results
    except Exception as e:
        print('error in getting links')
        return []


def scrape_and_combine(results_list):
    combined_text = ""
    
    for result in results_list:
        combined_text += f"Snippet: {result['snippet']}\n\n"
        
        try:
            response = requests.get(result['link'], timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            
            combined_text += f"Content: {text[:1000]}...\n\n"
            
        except:
            combined_text += "Content: Unable to scrape\n\n"
    
    return combined_text

