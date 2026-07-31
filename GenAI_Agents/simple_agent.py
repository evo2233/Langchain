import pandas as pd
import numpy as np
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def df_reader(df_path: str) -> str:
    """Read a CSV file and return the data as a markdown table."""
    return pd.read_csv(df_path).to_string()


agent = create_agent(
    model="ollama:qwen3:8b",  # 这个模型太笨了。。。最好还是用api
    tools=[df_reader],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
)

thread_config = {"configurable": {"thread_id": "1"}}


def answer_question(question):
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": question}]},
        config=thread_config,
        version="v3"
    )

    for message in stream.messages:
        for data in message.text:
            print(data, end="", flush=True)
        print()


def data_generator():
    n_rows = 1000

    # Define data categories
    makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
    models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
    colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

    data = {
        'Make': np.random.choice(makes, n_rows),
        'Model': np.random.choice(models, n_rows),
        'Color': np.random.choice(colors, n_rows),
        'Year': np.random.randint(2015, 2023, n_rows),
        'Price': np.random.uniform(20000, 80000, n_rows).round(2),
        'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
        'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
        'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
        'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
    }

    df = pd.DataFrame(data)
    df.to_csv("./cars.csv", index=False)


while True:
    text = input("Enter a question: ")
    if text == "q":
        break
    answer_question(text)
