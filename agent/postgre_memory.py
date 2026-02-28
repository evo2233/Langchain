from contextlib import contextmanager

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()


class CheckpointManager:
    _instance = None
    _pool = None

    def __init__(self, db_url):
        self.db_url = db_url
        self._pool = ConnectionPool(conninfo=db_url, max_size=5, kwargs={"autocommit": True})

    @classmethod
    def get_instance(cls, db_url):
        if cls._instance is None:
            cls._instance = cls(db_url)
        return cls._instance

    @contextmanager
    def get_checkpointer(self):
        with self._pool.connection() as conn:
            yield PostgresSaver(conn)

    def close(self):
        if self._pool:
            self._pool.close()


DB_URL = "postgresql://postgres:1Bian@localhost:5432/postgres?sslmode=disable"
manager = CheckpointManager.get_instance(DB_URL)


def ask_agent(query, thread_id="1"):
    with manager.get_checkpointer() as cp:
        # cp.setup()  # run this code only when update/init the config of DB to construct table
        agent = create_agent(
            model="deepseek-chat",
            checkpointer=cp
        )
        config = {"configurable": {"thread_id": thread_id}}

        results = agent.invoke(
            {"messages": {"role": "user", "content": query}},
            config=config
        )

        messages = results["messages"]
        for message in messages:
            message.pretty_print()


def get_latest_checkpoint(thread_id="1"):
    """
    checkpoint: A snapshot of a state graph
    XXXSaver: The manger of checkpoints
    """
    with manager.get_checkpointer() as cp:
        agent = create_agent(model="deepseek-chat", checkpointer=cp)
        config = {"configurable": {"thread_id": thread_id}}

        state_snapshot = agent.get_state(config)

        messages = state_snapshot.values.get("messages", [])

        if messages:
            print(f"the history record of thread {thread_id} is:")
            for message in messages:
                message.pretty_print()
        else:
            print(f"thread {thread_id} has no history record")


if __name__ == '__main__':
    try:
        chat_list = [
            # "Which city is Asakusa belongs to?",
            # "What about Akihabara?"
            "What we were talking about just now?"
        ]

        # for q in chat_list:
        #     ask_agent(q)

        # get_latest_checkpoint()
    finally:
        manager.close()
