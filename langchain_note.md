## 1. Agents



**你可以通过使用 Middleware 在运行时动态配置代理的系统提示词。**

在 `create_agent` 中，基础的 `system_prompt` 参数仅支持静态配置。若需要在运行时根据上下文、用户 ID 或其他业务逻辑灵活调整系统提示词，你应该实现并使用自定义 Middleware。

### 如何实现运行时系统提示词

Middleware 能够通过拦截代理循环中的消息或上下文，在每次模型调用前动态修改或注入系统提示词。要开始使用，你可以参考相关的 Middleware 概览文档来了解如何将自定义 Middleware 插入到 `create_agent` 的 `middleware` 参数列表中：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import BaseMiddleware

class DynamicSystemPromptMiddleware(BaseMiddleware):
    def __init__(self, prompt_generator):
        self.prompt_generator = prompt_generator

    def before_model_call(self, state):
        # 根据当前状态或上下文动态生成系统提示词
        new_prompt = self.prompt_generator(state)
        # 将生成的提示词注入到状态中
        state["system_prompt"] = new_prompt
        return state

# 在创建代理时挂载
agent = create_agent(
    model="...",
    tools=[...],
    middleware=[DynamicSystemPromptMiddleware(prompt_generator=my_logic)]
)
```

Middleware 是代理 harness 的核心扩展方式，它允许你以模块化的方式处理系统提示词、状态管理、上下文注入等需求。

**Relevant docs:**
- [Middleware 概览](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [自定义 Middleware 指南](https://docs.langchain.com/oss/python/langchain/middleware/custom)



> [!IMPORTANT]
>
> Persisting conversation history with `thread_id` requires the agent to be configured with a [checkpointer](https://docs.langchain.com/oss/python/langchain/long-term-memory). When deployed on [LangSmith](https://docs.langchain.com/langsmith/deployment), a checkpointer is provisioned automatically. Locally, pass one explicitly, for example `create_agent(..., checkpointer=InMemorySaver())`.

If you also need to pass per-run configuration (such as a user ID, API keys, or feature flags) to tools and middleware, pass it as `context` alongside `config`. Define the shape of that data with `context_schema` and access it through `runtime.context`:

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```



**As agents take on complex work, they need support across a few key areas. The middleware ecosystem provides**:

* **Execution environment** Tools, filesystem, sandboxes, and code execution

  ```python
  middleware=[FilesystemMiddleware(backend=StateBackend())]
  ```

* **Context management** Summarization, memory, skills, and prompt caching

  ```python
  SummarizationMiddleware(model=model, backend=backend)
  ```

* **Planning and delegation** Todo lists and subagents for parallel, isolated work

  ```python
  SubAgentMiddleware(
      backend=backend,
      subagents=[
          {
              "name": "researcher",
              "description": "Searches and returns a structured summary.",
              "system_prompt": "Use the search tool to research the question and summarize key points.",
              "tools": [search],
              "model": "anthropic:claude-sonnet-4-6",
              "middleware": [],
          }
      ],
  )
  ```

* **Fault tolerance** Retries, fallbacks, and call limits

  ```python
  ModelRetryMiddleware(max_retries=3),
  ToolRetryMiddleware(max_retries=2)
  ```

* **Guardrails** PII detection and content controls

  ```python
  PIIMiddleware("email")
  ```

* **Steering** Human-in-the-loop approval before high-impact actions

  ```python
  HumanInTheLoopMiddleware(interrupt_on={"write_file": True})
  ```

> [!IMPORTANT]
>
> The core agent loop involves calling a model, letting it choose tools to execute, and then finishing when it calls no more tools. Middleware exposes hooks before and after each of those steps. Before are some prebuild middleware, just add them to `middleware` list when `create_agent` they will work in the right time.  



**代理通过可插拔的 Backend（后端）与文件系统交互，这些后端不仅限于服务器存储。**

LangChain 的 Deep Agents 架构提供了一套统一的文件系统工具（如 `ls`, `read_file`, `write_file` 等），这些工具通过 Backend 协议与底层存储进行通信。这使得代理既可以与本地磁盘交互，也可以使用内存状态存储、数据库持久化存储或远程环境。

### 关于 Backend 的理解

你提到的 `backends` 文档中列出的内容涵盖了多种不同的场景，不仅仅是服务器上的文件存储：

*   **`StateBackend` (默认)**：这是与线程绑定的存储。文件被存储在 `langgraph` 的状态中，并随着对话线程（Thread）持久化，非常适合代理在执行过程中读写临时中间结果。
*   **`FilesystemBackend` (本地磁盘)**：这才是你所指的“本地文件系统交互”。通过设置 `root_dir`，你可以给代理指定本地机器上的某个目录，使其能够直接读写该目录下的真实文件。
*   **`StoreBackend` / `ContextHubBackend`**：这些提供跨线程、跨运行的持久化存储，适合作为代理的长期记忆或跨任务共享的数据仓库。
*   **沙箱环境 (Sandboxes)**：如果你需要安全地执行代码，可以使用沙箱（如 E2B、Modal 等）。这些环境通常自带一个隔离的文件系统，代理在其中进行的任何文件操作都不会影响你的主机。

### 如何安全地使用本地磁盘
若要让代理访问本地文件，请使用 `FilesystemBackend` 并指定绝对路径：

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# 注意：需谨慎授予代理本地磁盘访问权限
agent = create_deep_agent(
    model="...",
    backend=FilesystemBackend(root_dir="/absolute/path/to/your/files")
)
```

**建议：** 在开发阶段，你可以使用 `CompositeBackend`（复合后端）来构建路由，将 `/tmp/` 或 `/scratch/` 路由到 `StateBackend`（自动清理），而将需要处理的特定项目文件夹路由到 `FilesystemBackend`（本地磁盘），从而实现安全性与便捷性的平衡。

**Relevant docs:**

- [Backends 概览](https://docs.langchain.com/oss/python/deepagents/backends)
- [FilesystemBackend 参考](https://reference.langchain.com/python/deepagents/backends/filesystem/FilesystemBackend)





## 2. Models



#### Models can be utilized in two ways:

**With agents** - Models can be dynamically specified when creating an agent.

```python
agent = create_agent(model="google_genai:gemini-3.5-flash")
```

**Standalone** - Models can be called directly (outside of the agent loop) for tasks.

```python
import os
from langchain.chat_models import init_chat_model
os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-5.5")

response = model.invoke("Why do parrots talk?")
```

#### Key methods:

* **Invoke**: The model takes messages as input and outputs messages after generating a complete response.

  ```python
  from langchain.messages import HumanMessage, AIMessage, SystemMessage
  
  conversation = [
      SystemMessage("You are a helpful assistant that translates English to French."),
      HumanMessage("Translate: I love programming."),
      AIMessage("J'adore la programmation."),
      HumanMessage("Translate: I love building applications.")
  ]
  response = model.invoke(conversation)
  ```

* **Stream**: Invoke the model, but stream the output as it is generated in real-time.

  ```python
  for chunk in model.stream("What color is the sky?"):
      full = chunk if full is None else full + chunk
      print(full.text)
  
  # The
  # The sky
  # The sky is
  # The sky is typically
  # The sky is typically blue
  ```

* **Batch**: Send multiple requests to a model in a batch for more efficient processing.

  ```python
  responses = model.batch([
      "Why do parrots have colorful feathers?",
      "How do airplanes fly?",
      "What is quantum computing?"
  ])
  for response in responses:
      print(response)
  ```

  By default, `batch()` will only return the final output for the entire batch. If you want to receive the output for each individual input as it finishes generating, you can stream results with `batch_as_completed()`. When using `batch_as_completed()`, results may arrive **out of order**. Each includes the input index for matching to reconstruct the original order as needed.



> [!IMPORTANT]
>
> `max_tokens` 参数仅用于限制模型生成输出的 Token 数量，输入 Token 的数量通常由模型的上下文窗口限制（Context Window）来决定。

> [!CAUTION]
>
> LangChain chat models automatically retry failed API requests with exponential backoff. By default, models retry up to **6 times** for network errors, rate limits (429), and server errors (5xx). Client errors like 401 (unauthorized) or 404 are not retried.



#### Structured output 

`Pydantic` models provide the richest feature set with field validation, descriptions, and nested structures.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
```



#### Token usage

callback handler

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model(model="gpt-5.4-mini")
model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

callback = UsageMetadataCallbackHandler()
result_1 = model_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = model_2.invoke("Hello", config={"callbacks": [callback]})
print(callback.usage_metadata)
```

context manager

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback

model_1 = init_chat_model(model="gpt-5.4-mini")
model_2 = init_chat_model(model="claude-haiku-4-5-20251001")

with get_usage_metadata_callback() as cb:
    model_1.invoke("Hello")
    model_2.invoke("Hello")
    print(cb.usage_metadata)
```





## 3. Message



Messages are objects that contain:

-  **Role** - Identifies the message type (e.g. `system`, `user`)
-  **Content** - Represents the actual content of the message (like text, images, audio, documents, etc.)
-  **Metadata** - Optional fields such as response information, message IDs, and token usage



Text prompts

```python
response = model.invoke("Write a haiku about spring")
```

Message prompts

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
```



#### Message types

-  System message - Tells the model how to behave and provide context for interactions
-  Human message - Represents user input and interactions with the model
-  AI message - Responses generated by the model, including text content, tool calls, and metadata (like token usage)
  - AI message chunk - only in streaming, can be combined into a full message object 
-  Tool message - Represents the outputs of tool calls

> [!NOTE]
>
> Tool message 的存在是为了将工具执行的结果显式地传回给模型，它不会与 AI message 的内容重复。



**multimodal inputs** :

```python
from langchain.messages import HumanMessage

# String content
human_message = HumanMessage("Hello, how are you?")

# Provider-native format (e.g., OpenAI)
human_message = HumanMessage(content=[
    {"type": "text", "text": "What's in the picture?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# List of standard content blocks
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "What's in the picture?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
```





## 4. Tools



Type hints are **required** as they define the tool’s input schema. The docstring should be informative and concise to help the model understand the tool’s purpose.

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

> [!CAUTION]
>
> **Reserved argument names** :
>
> To access runtime information, use the [`ToolRuntime`](#`runtime = ToolRuntime`的基础用法) parameter instead of naming your own arguments `config` or `runtime`.
>
> ```python
> @tool
> def get_last_user_message(runtime: ToolRuntime) -> str:
> ```



**Customize tool properties** :

Override function name

```python
@tool("web_search")
def search(query: str) -> str:
```

Override doc-string

```python
@tool(description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
```



**Advanced schema definition** :

Use `Pydantic` as example

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```



#### `runtime = ToolRuntime`的基础用法

| parameter                                 |                                              |                |
| ----------------------------------------- | -------------------------------------------- | -------------- |
| `runtime.state.get("XXX")`                | get某state字段                               | state          |
| `return Command(update={})`               | update state 字段（[详见](#Tool execution)） | command        |
| `runtime.context.user_id`                 | get不可变config data                         | context        |
| `runtime.store`                           | 访问long-term memory                         | store          |
| `runtime.stream_writer`                   | 以流数据实时`print`给用户端                  | stream_writer  |
| `print(runtime.execution_info.thread_id)` | 访问thread ID, run ID和retry state等         | execution_info |
| `runtime.server_info`                     | 类似上条（仅适用LangGraph Server）           | server_info    |



#### Tool execution

tool 由 agent 调用，被 `ToolNode` 处理

* Tool return values：

  * return `string` 

  * return `object` (structured data)

  * return `command` (when you need to write to state)

    ```python
    from langchain.messages import ToolMessage
    from langchain.tools import ToolRuntime, tool
    from langgraph.types import Command
    
    
    @tool
    def set_language(language: str, runtime: ToolRuntime) -> Command:
        """Set the preferred response language."""
        return Command(
            update={
                "preferred_language": language,
                "messages": [
                    ToolMessage(
                        content=f"Language set to {language}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    ```

    > [!NOTE]
    >
    > Use reducers for fields that may be updated by parallel tool calls.

    > [!IMPORTANT]
    >
    > 设置为 `@tool(return_direct=True)` 的工具调用后会直接结束 agent loop

* error handling

  ```python
  from collections.abc import Callable
  
  from langchain.agents import create_agent
  from langchain.agents.middleware import wrap_tool_call
  from langchain.messages import ToolMessage
  from langchain.tools.tool_node import ToolCallRequest
  
  
  @wrap_tool_call
  def handle_tool_errors(
      request: ToolCallRequest,
      handler: Callable[[ToolCallRequest], ToolMessage],
  ) -> ToolMessage:
      """Convert tool exceptions into ToolMessages the model can handle."""
      try:
          return handler(request)
      except Exception as e:
          return ToolMessage(
              content=f"Tool error: Please check your input and try again. ({e})",
              tool_call_id=request.tool_call["id"],
          )
  
  
  agent = create_agent(
      model="google_genai:gemini-3.5-flash",
      tools=[],
      middleware=[handle_tool_errors],
  )
  ```



#### Dynamic tool selection

Too many tools may overwhelm the model (overload context) and increase errors; too few limit capabilities. Dynamic tool selection enables adapting the available toolset.

* Method 1: Filtering pre-registered tools
  1. register **middleware** `@wrap_model_call`, and pass in `request: ModelRequest`.
  2. set your **filter**.
  3. **override** `request = request.override(tools=tools)`.
* Method 2: Runtime tool registration (for Server)
  1. define `class DynamicToolMiddleware(AgentMiddleware)` mainly requires two middleware **hooks**, `wrap_model_call` and `wrap_tool_call`. (basic logic also use `override`)
  2. initial `middleware=[DynamicToolMiddleware()]`.



#### Headless tools

**仅在服务端注册了名称、描述和参数架构（Schema），但具体的执行逻辑完全运行在客户端（例如浏览器或其他环境）的工具。**实际上服务端会触发中断，将工具调用请求发送给客户端执行。客户端完成任务后，再将执行结果返回给服务端以恢复执行。例如：获取用户位置信息等。 



> [!NOTE]
>
> `LangChain` provides a large collection of prebuilt tools and toolkits for common tasks like web search, code interpretation, database access, and more. These ready-to-use tools can be directly integrated into your agents without writing custom code. See the [tools and toolkits](https://docs.langchain.com/oss/python/integrations/tools) integration page for a complete list of available tools organized by category.





## 5. Short-term memory

a short-term memory is the thread-level persistence, you need to specify a `checkpointer` when creating an agent.



**Save in memory**:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  


def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."


agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    thread_config,
)["messages"][-1].content

print(response)  # "Hi Bob! Nice to see you here. How are you doing?"

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    thread_config,
)["messages"][-1].content

print(response)  # "You are Bob!"
```

**Save in database**:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver  

def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgreSQL
    agent = create_agent(
        "gpt-5.5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```



#### Customizing agent memory

By default, agents use `message` filed in `AgentState` to store history, we can extend `AgentState` class to add more fields.

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5.5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})
```



#### Common patterns

* Trim messages: `@before_model` `RemoveMessage` 

  ```python
  @before_model
  def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
      """Keep only the last few messages to fit context window."""
      messages = state["messages"]
  
      if len(messages) <= 3:
          return None  # No changes needed
  
      first_msg = messages[0]
      recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
      new_messages = [first_msg] + recent_messages
  
      return {
          "messages": [
              RemoveMessage(id=REMOVE_ALL_MESSAGES),
              *new_messages
          ]
      }
  ```

* Delete messages: `after_model` `RemoveMessage` 

  ```python
  @after_model
  def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
  	return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}  # remove early 2 messages
  ```

* Summarize messages: [`SummarizationMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware#summarization) 

  ```python
  agent = create_agent(
      model="gpt-5.5",
      tools=[...],
      middleware=[
          SummarizationMiddleware(
              model="gpt-5.4-mini",
              trigger=("tokens", 4000),
              keep=("messages", 20)
          )
      ],
      checkpointer=checkpointer,
  )
  ```





## 6. Streaming



#### Event stream（v1.3+ & Fine grained）

Usually, use `stream_events()` to make the streaming output. (need `langchain` v1.3+)

```python
agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

stream = agent.stream_events({
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
}, version="v3")

for message in stream.messages:
    for delta in message.text:
        print(delta, end="", flush=True)
```

**Usage** of `stream_events`:

* `stream.messages`: Model message streams, one per LLM call.
  * `message.text`: Text deltas and final text for a message.
  * `message.reasoning`: Reasoning deltas for models that expose reasoning content.
  * `message.tool_calls`: Tool-call argument chunks and finalized tool calls.
* `stream.tool_calls`: Streams the lifecycle of tool execution after the tool call starts.
* `stream.values`: Access state snapshots.
* `stream.output`: Access the final agent state.
* `stream.subagents`: Subagent model message streams, same as stream.
  * `subagent.name`: Access the name of subagent.
  * `subagent.messages`: Similar with `stream.messages`.



#### Multiple projections 

**concurrent** consumption in async code, use `astream_events` with `asyncio.gather`:

```python
import asyncio

stream = await agent.astream_events(input, version="v3")

async def consume_messages():
    async for message in stream.messages:
        print(await message.text)

async def consume_tool_calls():
    async for call in stream.tool_calls:
        print(call.tool_name, call.input)

await asyncio.gather(consume_messages(), consume_tool_calls())
```

**synchronous** code, use `stream.interleave(...)` instead:

```python
stream = agent.stream_events(input, version="v3")

for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        print(item.text)
    elif name == "tool_calls":
        print(item.tool_name, item.input)
    elif name == "values":
        print(item)
```



#### Custom updates

Use custom stream transformers when your application needs a projection that is not built in, such as retrieval progress, artifacts, or domain-specific events.

An example for retrieval progress:

```python
from langgraph.config import get_stream_writer
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer

# 1. 在你的检索节点中发送进度事件(graph中的节点)
def retrieval_node(state):
    writer = get_stream_writer()
    # 发送自定义进度消息
    writer({"kind": "progress", "message": "正在检索相关文档..."})
    
    # ... 执行实际的检索逻辑 ...
    
    writer({"kind": "progress", "message": "检索完成，正在处理结果..."})
    return state

# 2. 定义处理这些事件的 Transformer
class RetrievalProgressTransformer(StreamTransformer):
    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        # 创建一个名为 "retrieval-progress" 的 channel
        self.progress_channel = StreamChannel("retrieval-progress")

    def init(self) -> dict:
        # 将 channel 暴露给流式结果
        return {"retrieval_progress": self.progress_channel}

    def process(self, event: ProtocolEvent) -> bool:
        # 捕获我们在节点中写入的自定义事件
        if event["method"] == "custom" and event["params"]["data"].get("kind") == "progress":
            self.progress_channel.push(event["params"]["data"]["message"])
        return True

# 3. 在运行流式处理时使用该 Transformer
stream = graph.stream_events(
    input_data, 
    version="v3", 
    transformers=[RetrievalProgressTransformer]
)

# 4. 在消费流时，从 extensions 中读取进度
for event in stream:
    if "retrieval_progress" in stream.extensions:
        # 你可以在这里处理进度更新
        print(f"进度更新: {stream.extensions['retrieval_progress'].values}")
```

✅**Code analyse**：

1. What's `StreamTransformer`?

   在 `LangGraph` 的事件流中，每个**事件**都以 `ProtocolEvent` 的类型传递。`StreamTransformer` 会对流中的每个 `ProtocolEvent` 进行处理，可用于监听、过滤、转换事件，并通过 `StreamChannel` 等机制将运行时信息暴露给外部消费者。

   > [!NOTE]
   >
   > **节点**（如 Retrieval、Tool、LLM）可通过 `get_stream_writer()` 产生 `custom` 类型的 `ProtocolEvent`，`StreamTransformer` 则负责捕获并处理这些事件。
   >
   > 后者可以通过控制`process`返回的bool值，实现过滤事件。

2. What's `StreamChannel`?

   在 `LangGraph` 中，Graph **State** 用于存储 Graph 运行过程中需要参与计算和传递的 State data (a part of runtime data)。对于检索进度、日志等仅用于 UI 展示的数据，如果也写入 Graph State，就会污染 State data。因此，`LangGraph` 提供了基于 **`StreamTransformer + StreamChannel`** 的 **Side Channel** 机制。节点可以通过 `get_stream_writer()` 发出自定义事件，Transformer 对这些事件进行处理，并将需要展示的数据写入 `StreamChannel`。消费者随后可以通过 `stream.extensions` 获取这些数据，用于更新 UI，而无需修改 Graph State。



#### Stream（v1.1+ & Easy way）

Pass one or more of the following **stream modes** as a list to the `stream` or `astream` methods:

* **updates**: 在 Graph 中的节点执行完成后，流式输出：(该节点的 name, 变更后的 state)。
*  **messages**: 流式传输大模型（LLM）的输出： (message_chunk, metadata)。
* **custom**: 在节点或工具内部手动流式传输**自定义数据**，通过 `langgraph.config.get_stream_writer` 写入。

stream modes 由 `chunk["type"]` 判断；内部数据由 `chunk["data"]` 读取，支持多种 modes 列表输入。

**Sub-agent distinction**: Through agent name is then available in metadata via the `lc_agent_name` key when streaming in `"messages"` mode. Don't forget to specify  `subgraphs=True` when creating the stream.

**Disable streaming**: Set `streaming=False` when initializing the model.





## Structured output



#### `response_format`

have 3 parameters: 

* `ProviderStrategy`: Uses **model-native structured output** capabilities.

  * `strict`: Optional bool parameter to enable strict schema adherence (`null` field is not allowed when it's True).

    ```python
    ProviderStrategy(FeedbackStrict, strict=True)
    ```

* `ToolStrategy`: Uses **tool-calling** to enforce structure. Wraps schema as a tool, call it to convert format.

* `type[StructuredResponseT]`: Directly pass your schema. `LangChain` will automatically selects the best strategy above.

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model="gpt-5.5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)
```



#### `ToolStrategy` parameters

**schema** (required): The schema defining the structured output format. Supports:

- **Pydantic models**: `BaseModel` subclasses with field validation. Returns validated Pydantic instance.
- **Dataclasses**: Python dataclasses with type annotations. Returns dict.
- **TypedDict**: Typed dictionary classes. Returns dict.
- **JSON Schema**: Dictionary with JSON schema specification. Returns dictionary.
- **Union types**: *Multiple schema options*. The model will choose the most appropriate schema based on the context.

> [!CAUTION]
>
> **Union types** is not supported in schema of `ProviderStrategy`.

**tool_message_content**: Custom content for the tool message returned when structured output is generated. If not provided, defaults to a message showing the structured response data.

**handle_errors**: Error handling strategy for structured output validation failures. Defaults to `True`.

- **`True`**: Catch all errors with default error template
- **`str`**: Catch all errors with this custom message
- **`type[Exception]`**: Only catch this exception type with default message
- **`tuple[type[Exception], ...]`**: Only catch these exception types with default message
- **`Callable[[Exception], str]`**: Custom function that returns error message
- **`False`**: No retry, let exceptions propagate
