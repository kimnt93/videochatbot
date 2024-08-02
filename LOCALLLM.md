# Local LLM: Analyzing the Cost of Deployment
*(Data 01/08/2024)*

**Source code**: [localllm](localllm)
## Table of Contents

1. [Self-Hosting LLM](#self-hosting-llm)
2. [Local LLM Evaluation](#llm-evaluation-metrics)
3. [LLM GPU Usage](#llm-gpu-usage)
4. [Analyzing LLM Cost: Self-Hosting LLM vs OpenAI, Claude and Gemini](#analyzing-llm-cost-self-hosting-cloud)
5. [Reducing LLM Usage Costs](#reducing-llm-usage-costs)
6. [About Self-Hosting LLM](#why-self-host-llm)

## Self-Hosting LLM

To deploy an on-premises LLM, we need to ensure the following requirements are met:
- **License of the LLM**: Ensure the model is commercially usable. For example, Llama2 is not for commercial use. Please check [Open LLMs](https://github.com/eugeneyan/open-llms).
- **Choosing the right LLM**: Select an LLM that fits your specific needs and use case. Ex, does the LLM support Korean and Thai lnguages, etc.?
- **Choosing the Embedding model**: Decide on the embedding model to be used alongside the LLM.
- **Framework to deploy**: Choose a suitable framework for deploying the LLM.
- **Choosing the right hardware**: Ensure you have the necessary hardware to support the LLM.
- **Instruction tuning**: Fine-tune the LLM with custom instructions to improve performance.

This example uses [litellm](https://github.com/BerriAI/litellm) to deploy LLM. The idea is to create a model endpoint by using [Ollama](https://ollama.com/), which is built on top of [llama-cpp](https://github.com/ggerganov/llama.cpp), and write custom text embedding. For text embedding, you can refer to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) or [sentence transformers](https://huggingface.co/sentence-transformers).

- The implementation of LLM inference and text embedding can be found at [models.py](localllm/app/models.py).

- The LLM and Embedding endpoint are formatted similarly to the OpenAI API. The implementation can be found at [router.py](localllm/app/router.py).

To define the configuration, follow the litellm routing config [documentation](https://docs.litellm.ai/docs/routing). For example, to deploy bloom-1b1 or bloomz-1b7:

```yaml
llm:
  - model_name: "gpt-3.5-turbo"
    litellm_params:
      model: "ollama/gouranshitera/bloom-1b1"
  - model_name: "gpt-3.5-turbo"
    litellm_params:
      model: "ollama/gouranshitera/bloomz-1b7"
```

and add an embedding model:

```yaml
- model_name: "local-e5-large-dim-1024"
  params:
    pretrained: "intfloat/multilingual-e5-small"
    prompt_template: "query: {query}" # prompt for retrieve task
    cuda: true
```

To start the API, run:

```sh
docker-compose up
```

Explore the project [localllm](localllm) that uses litellm to mimic the OpenAI API format. Refer to the OpenAI's API documentation [here](https://platform.openai.com/docs/api-reference/introduction) for more details and parameters.

#### Text Embedding Endpoint
```bash
curl http://0.0.0.0:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-e5-large-dim-1024",
    "input": "Hello world!"
  }'
```

#### Completions Endpoint
```bash
curl http://0.0.0.0:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt": "Say this is a test",
    "max_tokens": 50
  }'
```

#### Chat Completions Endpoint
```bash
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "What is the meaning of life?"}],
    "max_tokens": 50
  }'
```

#### Import OpenAI and Set Endpoint
```python
from openai import OpenAI

client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="empty")
```

#### Chat Completions Example
```python
chat = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?",
    },
  ],
)
print(chat.choices[0].message.content)
```

#### Completions Example
```python
completion = client.completions.create(
  model="gpt-3.5-turbo",
  prompt="Say this is a test",
)
print(completion.choices[0].text)
```

#### Text Embedding Example
```python
response = client.embeddings.create(
    input="Hello world!",
    model="local-e5-large-dim-1024"
)
print(response.data[0].embedding)
```

#### Langchain Examples

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


# Initialize LLM model
llm = ChatOpenAI(openai_api_base="http://0.0.0.0:8000/v1", openai_api_key="empty", model_name="gpt-3.5-turbo")
llm.invoke("What is the meaning of life?")


# Initialize Embedding model
embeddings = OpenAIEmbeddings(openai_api_base="http://0.0.0.0:8000/v1", openai_api_key="empty", model="local-e5-large-dim-1024")
embeddings.embed_query("Hello world!")
```


**Note**: You can see log probabilities and token probabilities in the response by adding param `echo=True` to the complete function. See the [litellm documentation](https://docs.litellm.ai/docs/providers/huggingface#viewing-log-probs). Here is output:
```json
{
  "id": "chatcmpl-3fc71792-c442-4ba1-a611-19dd0ac371ad",
  "object": "text_completion",
  "created": 1698801125.936519,
  "model": "bigcode/starcoder",
  "choices": [
    {
      "text": ", I'm going to make you a sand",
      "index": 0,
      "logprobs": {
        "tokens": ["good", " morning", ",", " I", "'m", " going", " to", " make", " you", " a", " s", "and"],
        "token_logprobs": ["None", -14.96875, -2.2285156, -2.734375, -2.0957031, -2.0917969, -0.09429932, -3.1132812, -1.3203125, -1.2304688, -1.6201172, -0.010292053]
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "completion_tokens": 9,
    "prompt_tokens": 2,
    "total_tokens": 11
  }
}
```

## LLM GPU Usage
The below table shows the GPU usage of LLMs (in GB), where Q4 is 4-bits quantization and F16 is 16-bits floating-point format. We must need +200% GPU memory with minimal hardware requirements to ensure the model runs smoothly. The longer context window requires more memory. Source: [GPU Benchmarks on LLM Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference).

| LLM           | Q4  | F16   |
|---------------|-----|-------|
| Llama3-8B     | 4.58  | 14.96 |
| Llama3-70B    | 39.59 | 131.42 |


## LLM Evaluation
LLM benchmarks on task and dataset can be found at [LLM Benchmark](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) or [ChatbotArena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard). More exposure to the LLM can be found at [The Big Benchmarks Collection
](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a).

## Analyzing LLM Cost: Self-Hosting Cloud

When analyzing the cost of deploying LLMs, both self-hosted and cloud-based, consider the following factors:

- **Comparison with Open LLM Service Costs**: Compare with services like OpenAI, Hugging Face, OpenRouter, VertexAI, Mixtral, etc.
- **Cost to Generate 1 Million Tokens**: Evaluate the cost based on the token generation capability of different models (e.g., OpenAI, Hugging Face, etc.).
- **Throughput and Latency**: Assess the throughput and latency of the LLM on various GPU hardware or cloud instances.

```
cost = <time to generate 1m tokens in hours> * <cost per hour>
```

Where:
```
time = 1m tokens / <throughput in tokens per second> / 3600
```

*Note: The cost of self-hosting LLM does not include prompt tokens.*

### Example: Performance and Cost Analysis

Here are the throughput tokens per seconds (TPS) for LLama3 on various hardware configurations using llama-cpp in Q4 and F16 formats, comparing LLama 8B and 70B models. Source: [GPU Benchmarks on LLM Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference).

| GPU                   | 8B Q4   | 8B F16  | 70B Q4  | 70B F16 |
|-----------------------|---------|---------|---------|---------|
| RTX A6000 48GB        | 102.22  | 40.25   | 14.58   | _       |
| A100 PCIe 80GB        | 138.31  | 54.56   | 22.11   | _       |
| H100 PCIe 80GB        | 144.49  | 67.79   | 25.01   | _       |
| 4x A100 PCIe 80GB     | 117.30  | 51.54   | 22.68   | 7.38    |
| 4x H100 PCIe 80GB     | 118.14  | 62.90   | 26.20   | 9.63    |

#### Example Costs from Lambda Labs
Source: https://lambdalabs.com/service/gpu-cloud

| GPUs                   | VRAM / GPU | vCPUs | RAM      | STORAGE      | PRICE*        |
|------------------------|------------|-------|----------|--------------|---------------|
| 1x NVIDIA A6000 48 GB  | 48 GB      | 14    | 100 GiB  | 200 GiB SSD  | $0.80 / hr    |
| 1x NVIDIA A100 PCIe 80 GB | 80 GB      | 30    | 200 GiB  | 512 GiB SSD  | $1.89 / hr    |
| 1x NVIDIA H100 PCIe 80 GB | 80 GB      | 26    | 200 GiB  | 1 TiB SSD    | $2.49 / hr    |
| 4x NVIDIA A100 SXM 80 GB | 80 GB      | 240   | 1800 GiB | 20 TiB SSD   | $7.22 / hr    |
| 4x NVIDIA H100 SXM 80 GB | 80 GB      | 208   | 1800 GiB | 26 TiB SSD   | $16.92 / hr   |

#### Calculated Costs (USD)

| GPU                   | 8B Q4 | 8B F16 | 70B Q4 | 70B F16 |
|-----------------------|-------|--------|--------|---------|
| RTX A6000 48GB        | 2.17  | 5.52   | 14.58  | _       |
| A100 PCIe 80GB        | 3.79  | 9.62   | 23.74  | _       |
| H100 PCIe 80GB        | 4.78  | 10.20  | 27.65  | _       |
| 4x A100 PCIe 80GB     | 17.09 | 38.91  | 88.42  | 71.75   |
| 4x H100 PCIe 80GB     | 16.97 | 74.72  | 179.38 | 488.05  |

### Comparing with OpenAI, Gemini, Claude, etc.

For RAG (Retrieval-Augmented Generation) applications, assume the cost to generate 1 million tokens includes [12m input + 1m output]. Note that the tokenizers of different models vary. For instance, 1 million tokens of GPT-3.5-turbo approximately equals 700k tokens of Llama3-8B.

*Prices from [OpenRouter](https://openrouter.ai/models)*

| LLM                      | Price for 1M tokens | TPS    |
|--------------------------|---------------------|--------|
| OpenAI/GPT-3.5-turbo     | $7.5                | 85.56  |
| OpenAI/GPT-3.5-turbo-16k | $9                  | 92.10  |
| OpenAI/GPT-4o-mini       | $2.4                | 74.05  |
| Anthropic/Claude-3-haiku | $4.25               | 145.92 |
| Google/Gemini-flash-1.5  | $3.75               | 144.77 |
| GROQ/Llama3-8B           | $0.68               | 1250   |
| GROQ/Llama3-70B          | $7.51               | 330    |
| Llama-3-8B Q4            | $2.17               | 102.22 |

For the above table, if we self-host Llama3-8B and compare with GPT-3.5, we must use at least 30% running time of the cloud instance to be cost-effective.

## Reducing LLM Usage Costs

- **Route queries**: If the query is simple, use a simpler or cheaper model. An example implementation.model can be fount at [RouteLLM](https://github.com/lm-sys/RouteLLM).
- **Take advantage of free requests**: Some services offer free requests, use them. Examples include Groq (https://console.groq.com/settings/billing) and Gemini (https://ai.google.dev/pricing).
- **Route LLM**: Randomly route queries to different models to balance the load and cost (https://github.com/BerriAI/litellm).

## Why Self-Host LLM?
Using an LLM API service can be cheaper than self-hosting, but self-hosting has several advantages:

- **Privacy**: Keep your data private and secure, as your information does not leave your infrastructure.
- **Customization**: Customize the model to your specific needs with custom instructions, embeddings, longer context length, and fine-tuning.
- **Avoid Vendor Lock-in**: Maintain flexibility and control over your infrastructure and avoid being dependent on vendor's ecosystem.

While self-hosting also has its challenges, such as maintenance, scalability, and cost. It is essential to weigh the pros and cons before deciding to self-host an LLM.

You can follow the Reddit discussions:
- [whether local LLM is cheaper than ChatGPT API](https://www.reddit.com/r/LocalLLaMA/comments/13pt5f3/is_local_llm_cheaper_than_chatgpt_api/)
- [whether running LLM locally or executing is cheaper](https://www.reddit.com/r/ollama/comments/1dwr1oi/which_is_cheaper_running_llm_locally_or_executing/)
- [the most cost-effective way to scale up](https://www.reddit.com/r/LocalLLaMA/comments/1bvpbbe/whats_the_most_cost_effective_way_to_scale_up/)