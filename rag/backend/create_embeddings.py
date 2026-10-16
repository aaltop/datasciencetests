import src.env

from src.models import sentence_transformer
from src.filesystem import fs
from src.processing import encode
        
    

def main():
    model = sentence_transformer()
    encode(sorted(fs.text_data.rglob("*.txt")), name="python_docs_text", model=model)
    

if __name__ == "__main__":
   main()