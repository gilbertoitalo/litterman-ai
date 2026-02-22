from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# print(f"Chave carregada: {api_key[:8] if api_key else 'VAZIA - problema no .env'}")
# print(f"Chave completa: {api_key}")   # Carrega as variáveis de ambiente do arquivo .env

# Substitua pela sua chave ou configure como variável de ambiente
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents="Olá Gemini! Diga 'Olá Mundo' de uma forma criativa."
)

print(response.text)