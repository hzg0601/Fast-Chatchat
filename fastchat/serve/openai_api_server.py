"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

Usage:
python3 -m fastchat.serve.openai_api_server

各种检查函数:
check_api_key, check_model,check_length,check_requests

各种处理函数：
process_input: 
将Union[str,List[str],List[List[str]]]转换为List[str]

get_worker_address:
# 根据model_name调用httpx.AsyncClient访问controller_address + "/get_worker_address"
# 得到model_name对应的worker的地址

get_conv:
# 1. 调用get_worker_address函数获取模型对应的worker_addr,
# 2. 根据worker_addr，model_name在全局的conv_template_map找到对话模板conv_template
# 3. 如果全局的conv_template_map没有对应模板，则
#    基于httpx.AsyncClient执行post，访问worker_addr + "/worker_get_conv_template"获取模板，
#    然后加入到全局的conv_template_map中。

get_gen_params:
# 1. 调用get_conv函数根据model_name在全局的conv_template_map或
#    调用get_worker_addr访问worker_addr + "/worker_get_conv_template"获取模板
# 2. 调用conversation.py的Conversation类将模板信息抽出构造为一个Conversation实例
# 3. 根据message的role字段更新Conversation实例，完成后调用实例的get_prompt方法得到最终的prompt
# 4. 根据各关键词构造gen_params，更新停止词，返回gen_params.

# 对话用函数
chat_completion_stream_generator:
# 1. 调用fastchat.protocol.openai_api_protocol.ChatCompletionResponseStreamChoice构造choice_data,
# 2. 基于choice_data调用fastchat.protocol.openai_api_protocol.ChatCompletionStreamResponse构造chunk
# 3. 调用generate_completion_stream访问对应worker的generate_stream_gate方法得到一个生成器response，
#    response.aiter_raw方法yield内容，# 
# 4. 如果yield的输出无误，则正常返回
# 5. 如果有误，则找出新增文本,  用CompletionResponseStreamChoice，CompletionStreamResponse组装新增文本，
#    如果新增文本不为空，则yield结果，如果新增文本为空，则将组装数据计入到finish_stream_events并最终yield结果。
# 6. 最后yield "data: [DONE]\n\n".

generate_completion_stream_generator：
与chat_completion_stream_generator几乎完全一致。

generate_completion_stream：
# 1. 调用get_worker_address访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr + "/worker_generate_stream",
#    该访问会调用model_worker.py对应worker的generate_stream_gate方法得到一个生成器response
# 3. response.aiter_raw方法yield内容

generate_completion：
# 1. 调用get_worker_address，访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr + "/worker_generate"得到输出

get_embedding:
# 1. 调用get_worker_address访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr + "/worker_get_embedding"调用model_worker.py的函数得到输出

接口函数：
'v1/models'(show_available_models):
# 使用httpx.AsyncClient()访问controller_address + "/refresh_all_workers"更新模型
# 访问controller_address + "/list_models"列出模型
# 以fastchat.protocol.openai_api_protocol.ModelList(ModelCard)的形式返回

'/v1/chat/completions'(create_chat_completion)
# 1. 调用check_mdoel检查模型,check_requests检查requests,get_gen_params构造生成参数，check_length检查文本长度
# 2. 如果request.stream为True,则调用chat_completion_stream_generator，得到输出的生成器，以生成器初始化StreamingResponse返回结果
# 3. 如果不是stream模式，则以asyncio.create_task(generate_completion(gen_params))得到所有结果，调用asyncio.gather获取全部对话
#    遍历全部对话，以对话内容为主体构造ChatCompletionResponseChoice实例，装入choices中，用ChatCompletionResponse封装choices并返回。

"/v1/completions"（create_completion）
# 与`chat/completions`几乎一致，只是：
# 1. gen_params的位置放在request.stream判断之后，
# 2. CompletionResponseChoice的构造使用了参数logprobs=content.get("logprobs", None)
# 3. usage 改为了UsageInfo.parse_obj(usage)

"/v1/embeddings"（create_embedding）
"/v1/engines/{model_name}/embeddings"（create_embedding）

# 1. 执行check_model, process_input等处理和检查,调用get_embedding函数
#   访问worker_addr + "/worker_get_embedding"调用model_worker.py的函数得到输出
# 2. 如果返回错误，执行错误处理；否则，调用EmbeddingsResponse返回结果

"/api/v1/token_check"(count_tokens):
# 1. 调用get_worker_address访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr +model_details/count_token得到context_length、count
# 3. 判断输入token是否超限，返回APITokenCheckResponse类。

"/api/v1/chat/completions"(create_chat_completion)
# 与'chat/completions'完全一致


"""
import asyncio
import argparse
import asyncio
import json
import logging
import os
from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi import Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from pydantic import BaseSettings
import shortuuid
import tiktoken
import uvicorn

from fastchat.constants import (
    WORKER_API_TIMEOUT,
    WORKER_API_EMBEDDING_BATCH_SIZE,
    ErrorCode,
)
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from fastapi.exceptions import RequestValidationError
# openai_api返回类的格式
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)
# 基本api的返回格式
from fastchat.protocol.api_protocol import (
    APIChatCompletionRequest,
    APITokenCheckRequest,
    APITokenCheckResponse,
    APITokenCheckResponseItem,
)

logger = logging.getLogger(__name__)

conv_template_map = {}

# Base class for settings, allowing values to be overridden by environment variables.
# This is useful in production for secrets you do not wish to save in code, 
# it plays nicely with docker(-compose), Heroku and any 12 factor app design.
# 允许值被环境变量覆盖的settings
# pydantic的基本类
class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    api_keys: List[str] = None


app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}
# http的认证方式之一Bearer,在Bearer认证中的凭证称为Bearer_token 或者Access_token。
# 该种方式的优点就是灵活方便，因为凭证的生成和验证完全由开发人员设计和实现。
# 目前最流行的token编码协议就是JWT(JSON WEB TOKEN)
# http认证根据凭证协议的不同，划分为不同的方式。常用的方式有：HTTP基本认证,HTTP摘要认证,HTTP Bearer认证
get_bearer_token = HTTPBearer(auto_error=False)

# Depends() is what FastAPI will actually use to know what is the dependency.
# From it is that FastAPI will extract the declared parameters and that is what FastAPI will actually call.
# Depends先执行对应的函数，然后返回结果

# 检查AppSettings().api_keys列表中是否不为空，参数auth是否不为空，auth.credentials是否在全局的AppSettings().api_keys列表中，
# 若AppSettings().api_keys列表中不为空，且检查通过，返回token,若不通过抛出错误，若列表为空，则返回None
#? AppSettings内定义的属性是类的属性不是实例的属性啊？
async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        # 海象运算符：将表达式的值赋值给变量，然后返回表达式的值
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None

# 这种函数的意义在哪里？
def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )

# 校验错误的handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))

# 调用get_worker_address测试model对应的worker是否能访问，
# 如能访问返回None, 否则返回错误信息
async def check_model(request) -> Optional[JSONResponse]:
    controller_address = app_settings.controller_address
    ret = None
    async with httpx.AsyncClient() as client:
        try:
            _worker_addr = await get_worker_address(request.model, client)
        except:
            models_ret = await client.post(controller_address + "/list_models")
            models = models_ret.json()["models"]
            ret = create_error_response(
                ErrorCode.INVALID_MODEL,
                f"Only {'&&'.join(models)} allowed now, your model {request.model}",
            )
    return ret

# 检查输入的prompt的token数+max_tokens是否超出context_length
# context_length为输入文本长度限制，即max_sequence_length或类似参数;
#? 为什么额外加一个max_tokens参数？如何确定
async def check_length(request, prompt, max_tokens):
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(request.model, client)

        response = await client.post(
            worker_addr + "/model_details",
            headers=headers,
            json={"model": request.model},
            timeout=WORKER_API_TIMEOUT,
        )
        context_len = response.json()["context_length"]

        response = await client.post(
            worker_addr + "/count_token",
            headers=headers,
            json={"model": request.model, "prompt": prompt},
            timeout=WORKER_API_TIMEOUT,
        )
        token_num = response.json()["count"]

    if token_num + max_tokens > context_len:
        return create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None

# 检查request的各参数是否满足要求
def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None

# tiktoken是OpenAI开源的一个快速分词工具。它将一个文本字符串（例如“tiktoken很棒！”）和一个编码（例如“cl100k_base”）作为输入，
# 然后将字符串拆分为标记列表（例如["t"，"ik"，"token"，" is"，" great"，"!"]）。

# input的处理函数
# 1. 如果是str,作为列表直接返回
# 2. 如果是List[int],调用tiktoken.encoding_for_model(model_name)检索模型的编码模式，得到一个编码模型
#    编码模型对数字仅解码，作为一个str的列表返回
# 3. 如果是List[List[int]],对每个子list进行解码得到一个str的列表返回
def process_input(model_name:str, inp:Union[str,List[str],List[List[str]]])-> List[str]:
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(text) for text in inp]

    return inp

# 1. 调用get_conv函数根据model_name在全局的conv_template_map或
#    调用get_worker_addr访问worker_addr + "/worker_get_conv_template"获取模板
# 2. 调用conversation.py的Conversation类将模板信息抽出构造为一个Conversation实例
# 3. 根据message的role字段更新Conversation实例，完成后调用实例的get_prompt方法得到最终的prompt
# 4. 根据各关键词构造gen_params，更新停止词，返回gen_params.
async def get_gen_params(
    model_name: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]],
) -> Dict[str, Any]:
    conv = await get_conv(model_name)
    conv = Conversation(
        name=conv["name"],
        system=conv["system"],
        roles=conv["roles"],
        messages=list(conv["messages"]),  # prevent in-place modification
        offset=conv["offset"],
        sep_style=SeparatorStyle(conv["sep_style"]),
        sep=conv["sep"],
        sep2=conv["sep2"],
        stop_str=conv["stop_str"],
        stop_token_ids=conv["stop_token_ids"],
    )

    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    if max_tokens is None:
        max_tokens = 512
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
    }

    if not stop:
        gen_params.update(
            {"stop": conv.stop_str, "stop_token_ids": conv.stop_token_ids}
        )
    else:
        gen_params.update({"stop": stop})

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params

# 1.根据model_name调用httpx.AsyncClient访问controller_address + "/get_worker_address"
#   得到model_name对应的worker的地址
async def get_worker_address(model_name: str, client: httpx.AsyncClient) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :param client: The httpx client to use
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address

    ret = await client.post(
        controller_address + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")

    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr

# httpx.AsyncClient: An asynchronous HTTP client, with connection pooling, HTTP/2, redirects, cookie persistence, etc.
# 1. 调用get_worker_address函数获取模型对应的worker_addr,
# 2. 根据worker_addr，model_name在全局的conv_template_map找到对话模板conv_template
# 3. 如果全局的conv_template_map没有对应模板，则
#    基于httpx.AsyncClient执行post，访问worker_addr + "/worker_get_conv_template"获取模板，
#    然后加入到全局的conv_template_map中。
async def get_conv(model_name: str):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(model_name, client)
        conv_template = conv_template_map.get((worker_addr, model_name))
        if conv_template is None:
            response = await client.post(
                worker_addr + "/worker_get_conv_template",
                headers=headers,
                json={"model": model_name},
                timeout=WORKER_API_TIMEOUT,
            )
            conv_template = response.json()["conv"]
            conv_template_map[(worker_addr, model_name)] = conv_template
        return conv_template

# 使用httpx.AsyncClient()访问controller_address + "/refresh_all_workers"更新模型
# 访问controller_address + "/list_models"列出模型
# 以fastchat.protocol.openai_api_protocol.ModelList(ModelCard)的形式返回
@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        ret = await client.post(controller_address + "/refresh_all_workers")
        ret = await client.post(controller_address + "/list_models")
    models = ret.json()["models"]
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)

# 1. 调用check_mdoel检查模型,check_requests检查requests,get_gen_params构造生成参数，check_length检查文本长度
# 2. 如果request.stream为True,则调用chat_completion_stream_generator，得到输出的生成器，以生成器初始化StreamingResponse返回结果
# 3. 如果不是stream模式，则以asyncio.create_task(generate_completion(gen_params))得到所有结果，调用asyncio.gather获取全部对话
#    遍历全部对话，以对话内容为主体构造ChatCompletionResponseChoice实例，装入choices中，用ChatCompletionResponse封装choices并返回。
@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    gen_params = await get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
    )
    error_check_ret = await check_length(
        request, gen_params["prompt"], gen_params["max_new_tokens"]
    )
    if error_check_ret is not None:
        return error_check_ret

    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(generate_completion(gen_params))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )
        if "usage" in content:
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

# 1. 调用fastchat.protocol.openai_api_protocol.ChatCompletionResponseStreamChoice构造choice_data,
# 2. 基于choice_data调用fastchat.protocol.openai_api_protocol.ChatCompletionStreamResponse构造chunk
# 3. 调用generate_completion_stream访问对应worker的generate_stream_gate方法得到一个生成器response，
#    response.aiter_raw方法yield内容，# 
# 4. 如果yield的输出无误，则正常返回
# 5. 如果有误，则找出新增文本,  用CompletionResponseStreamChoice，CompletionStreamResponse组装新增文本，
#    如果新增文本不为空，则yield结果，如果新增文本为空，则将组装数据计入到finish_stream_events并最终yield结果。
# 6. 最后yield "data: [DONE]\n\n".

#* 与generate_completion_stream_generator几乎完全一致。
async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(gen_params):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=model_name
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

# 与`chat/completions`基本一致，只是：
# 1. gen_params的位置放在request.stream判断之后，
# 2. CompletionResponseChoice的构造使用了参数logprobs=content.get("logprobs", None)
# 3. usage 改为了UsageInfo.parse_obj(usage)
@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    request.prompt = process_input(request.model, request.prompt)

    for text in request.prompt:
        error_check_ret = await check_length(request, text, request.max_tokens)
        if error_check_ret is not None:
            return error_check_ret

    if request.stream:
        generator = generate_completion_stream_generator(request, request.n)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for text in request.prompt:
            gen_params = await get_gen_params(
                request.model,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                echo=request.echo,
                stream=request.stream,
                stop=request.stop,
            )
            for i in range(request.n):
                content = asyncio.create_task(generate_completion(gen_params))
                text_completions.append(content)

        try:
            all_tasks = await asyncio.gather(*text_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

        choices = []
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                CompletionResponseChoice(
                    index=i,
                    text=content["text"],
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
        )

# 1. 根据request的prompt字段调用get_gen_params构造生成参数
# 2. 调用generate_completion_stream函数，该函数会最终访问worker_addr + "/worker_generate_stream"
#    调用model_worker.py对应worker的generate_stream_gate方法得到一个生成器response并yield输出
# 3. 如果yield的输出无误，则正常返回
# 4. 如果有误，则找出新增文本,  用CompletionResponseStreamChoice，CompletionStreamResponse组装新增文本，
#    如果新增文本不为空，则yield结果，如果新增文本为空，则将组装数据计入到finish_stream_events并最终yield结果。
# 5. 最后yield "data: [DONE]\n\n".
async def generate_completion_stream_generator(request: CompletionRequest, n: int):
    model_name = request.model
    id = f"cmpl-{shortuuid.random()}"
    finish_stream_events = []
    for text in request.prompt:
        for i in range(n):
            previous_text = ""
            gen_params = await get_gen_params(
                request.model,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                echo=request.echo,
                stream=request.stream,
                stop=request.stop,
            )
            async for content in generate_completion_stream(gen_params):
                if content["error_code"] != 0:
                    yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text) :]
                previous_text = decoded_unicode
                # todo: index is not apparent
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text=delta_text,
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", None),
                )
                chunk = CompletionStreamResponse(
                    id=id,
                    object="text_completion",
                    choices=[choice_data],
                    model=model_name,
                )
                if len(delta_text) == 0:
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

# 1. 调用get_worker_address访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr + "/worker_generate_stream",
#    该访问会调用model_worker.py对应worker的generate_stream_gate方法得到一个生成器response
# 3. response.aiter_raw方法yield内容
async def generate_completion_stream(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(payload["model"], client)
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        ) as response:
            # content = await response.aread()
            async for raw_chunk in response.aiter_raw():
                for chunk in raw_chunk.split(delimiter):
                    if not chunk:
                        continue
                    data = json.loads(chunk.decode())
                    yield data

# 1. 调用get_worker_address，访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr + "/worker_generate"得到输出
async def generate_completion(payload: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(payload["model"], client)

        response = await client.post(
            worker_addr + "/worker_generate",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        completion = response.json()
        return completion

# 两个接口同一个函数
# 1. 执行check_model, process_input等处理和检查,调用get_embedding函数
#   访问worker_addr + "/worker_get_embedding"调用model_worker.py的函数得到输出
# 2. 如果返回错误，执行错误处理；否则，调用EmbeddingsResponse返回结果
@app.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
@app.post("/v1/engines/{model_name}/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    request.input = process_input(request.model, request.input)

    data = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.input[i : min(i + batch_size, len(request.input))]
        for i in range(0, len(request.input), batch_size)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
        }
        embedding = await get_embedding(payload)
        if "error_code" in embedding and embedding["error_code"] != 0:
            return create_error_response(embedding["error_code"], embedding["text"])
        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embedding["embedding"])
        ]
        token_num += embedding["token_num"]
    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)

# 1. 调用get_worker_address访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr + "/worker_get_embedding"调用model_worker.py的函数得到输出
async def get_embedding(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    model_name = payload["model"]
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(model_name, client)

        response = await client.post(
            worker_addr + "/worker_get_embeddings",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        embedding = response.json()
        return embedding


### GENERAL API - NOT OPENAI COMPATIBLE ###

# 1. 调用get_worker_address访问controller_address + "/get_worker_address"得到model对应的地址
# 2. 根据地址调用httpx.AsyncClient.stream，访问worker_addr +model_details/count_token得到context_length、count
# 3. 判断输入token是否超限，返回APITokenCheckResponse类。
@app.post("/api/v1/token_check")
async def count_tokens(request: APITokenCheckRequest):
    """
    Checks the token count for each message in your list
    This is not part of the OpenAI API spec.
    """
    checkedList = []
    async with httpx.AsyncClient() as client:
        for item in request.prompts:
            worker_addr = await get_worker_address(item.model, client)

            response = await client.post(
                worker_addr + "/model_details",
                headers=headers,
                json={"model": item.model},
                timeout=WORKER_API_TIMEOUT,
            )
            context_len = response.json()["context_length"]

            response = await client.post(
                worker_addr + "/count_token",
                headers=headers,
                json={"prompt": item.prompt, "model": item.model},
                timeout=WORKER_API_TIMEOUT,
            )
            token_num = response.json()["count"]

            can_fit = True
            if token_num + item.max_tokens > context_len:
                can_fit = False

            checkedList.append(
                APITokenCheckResponseItem(
                    fits=can_fit, contextLength=context_len, tokenCount=token_num
                )
            )

    return APITokenCheckResponse(prompts=checkedList)

# 与chat/completions应完全一致
@app.post("/api/v1/chat/completions")
async def create_chat_completion(request: APIChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    gen_params = await get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        stop=request.stop,
    )

    if request.repetition_penalty is not None:
        gen_params["repetition_penalty"] = request.repetition_penalty

    error_check_ret = await check_length(
        request, gen_params["prompt"], gen_params["max_new_tokens"]
    )
    if error_check_ret is not None:
        return error_check_ret

    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(generate_completion(gen_params))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )
        task_usage = UsageInfo.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


### END GENERAL API - NOT OPENAI COMPATIBLE ###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-keys",
        type=lambda s: s.split(","),
        help="Optional list of comma separated API keys",
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.controller_address = args.controller_address
    app_settings.api_keys = args.api_keys

    logger.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
