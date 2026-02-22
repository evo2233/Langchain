from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    temperature=0.1
)

for chunk in model.stream("介绍一下JANE DOE"):
    print(chunk.content, flush=True)
