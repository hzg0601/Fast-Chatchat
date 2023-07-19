"""
A model worker that executes the model.
本脚本中所有post接口的输入都是Request类，该类由具体的模型接口提供
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import List
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from starlette.responses import RedirectResponse
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import torch.nn.functional as F
import uvicorn

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.utils import build_logger, pretty_print_semaphore, get_context_length


worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

app = FastAPI()


def heart_beat_worker(obj):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        obj.send_heart_beat()

# worker的基类
# 1. 定义init_heart_beat：先执行register_to_controller检查是否能访问{worker_name}+/worker_get_status
#    然后以多线程执行heart_beat_worker,heart_beat_worker访问controller_addr + "/receive_heart_beat"，将worker_addr作为参数输入，
#   调用controller.receive_heart_beat，receive_heart_beat更新queue_length,并以当前时间为last_heart_beat

# 2. 定义register_to_controller：访问controller_addr +"/register_worker"接口，调用controller.register_worker方法，
#    register_worker方法调用controller.get_worker_status访问worker_name + "/worker_get_status"，

# 3. 定义send_heart_beat: 访问controller_addr + "/receive_heart_beat"，将worker_addr作为参数输入，
#    调用controller.receive_heart_beat，receive_heart_beat更新queue_length,并以当前时间为last_heart_beat
#    若访问成功，则结束循环，否则抛出异常；
# 4. get_queue_length,get_conv_template,get_status,count_token等更新模型执行相关信息。
class BaseModelWorker:
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        # model_path若为本地路径需要以"/"为结尾
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or [model_path.split("/")[-1]]
        self.limit_worker_concurrency = limit_worker_concurrency

        self.conv = get_conversation_template(model_path)
        self.tokenizer = None
        self.context_len = None
        self.call_ct = 0
        self.semaphore = None

        self.heart_beat_thread = None
    # 先执行register_to_controller检查是否能访问{worker_name}+/worker_get_status
    # 然后以多线程执行heart_beat_worker
    def init_heart_beat(self):
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,)
        )
        self.heart_beat_thread.start()

    # 访问controller_addr +"/register_worker"接口，调用controller.register_worker方法，而controller.register_worker
    # 以worker_addr作为数据输出，其调用controller.get_worker_status方法，执行requests.post方法访问{worker_name}+/worker_get_status
    #? 还是没有定义worker各接口的方法啊
    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200
    
    # 访问controller_addr + "/receive_heart_beat"，将worker_addr作为参数输入，
    # 接口会调用controller.receive_heart_beat，更新queue_length,并以当前时间为last_heart_beat
    # 若访问成功，则结束循环，否则抛出异常
    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(self.semaphore)}. "
            f"call_ct: {self.call_ct}. "
            f"worker_id: {self.worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()
    # queue_length = limit_worker_concurrency - semaphore._value + semaphore._waiters
    #? queue_length 是并发时等待队列的长度
    def get_queue_length(self):
        if (
            self.semaphore is None
            or self.semaphore._value is None
            or self.semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.limit_worker_concurrency
                - self.semaphore._value
                + len(self.semaphore._waiters)
            )
    #? speed是什么？
    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def count_token(self, params):
        prompt = params["prompt"]
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def get_conv_template(self):
        return {"conv": self.conv}

# worker的类
# 1. __init__: 初始化基类,调用model_adapter.py load_model加载模型和tokenizer;
#               获取输入文本长度限制，即max_sequence_length或类似参数;
#               调用model_adapter.py的get_generate_stream_function方法获取generate_stream_func方法
# 2. generate_stream_gate: 调用generate_stream_func方法，执行流式输出，并执行各种错误处理。
#                           prompt需要包含在输入的param。
# 3. generate_gate: 调用generate_stream_gate，但不返回最后一次的输出；
# 4. get_embeddings:调用tokenizer.batch_encode_plus方法得到输入的encoding, 取出input_ids, attention_mask
#                   调用model的forward方法得到输出，以输出.hidden_states的最后一层为初始embedding记为data,
#                   以data * mask 然后执行求和、按seq_len标准化、L2标准化得到最终的embedding.
# 5. 
class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: bool = None,
        stream_interval: int = 2,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.model, self.tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            load_8bit,
            cpu_offloading,
            gptq_config,
        )
        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval

        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        self.call_ct += 1

        try:
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())
    #   调用tokenizer.batch_encode_plus方法得到输入的encoding, 取出input_ids, attention_mask
    #   调用model的forward方法得到输出，以输出.hidden_states的最后一层为初始embedding记为data,
    #   以data * mask 然后执行求和、按seq_len标准化、L2标准化得到最终的embedding.
    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            tokenizer = self.tokenizer
            is_llama = "llama" in str(
                type(self.model)
            )  # llama supports batch inference
            is_chatglm = "chatglm" in str(type(self.model))
            is_t5 = "t5" in str(type(self.model))
            if is_llama:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                model_output = self.model(
                    input_ids, attention_mask, output_hidden_states=True
                )
                data = model_output.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
                masked_embeddings = data * mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                seq_length = torch.sum(mask, dim=1)
                embedding = sum_embeddings / seq_length
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret = {
                    "embedding": normalized_embeddings.tolist(),
                    "token_num": torch.sum(attention_mask).item(),
                }
            else:
                embedding = []
                token_num = 0
                for text in params["input"]:
                    input_ids = tokenizer.encode(text, return_tensors="pt").to(
                        self.device
                    )
                    if is_t5:
                        model_output = self.model(
                            input_ids, decoder_input_ids=input_ids
                        )
                    else:
                        model_output = self.model(input_ids, output_hidden_states=True)
                    if is_chatglm:
                        data = (model_output.hidden_states[-1].transpose(0, 1))[0]
                    elif is_t5:
                        data = model_output.encoder_last_hidden_state[0]
                    else:
                        data = model_output.hidden_states[-1][0]
                    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                    embedding.append(data.tolist())
                    token_num += len(input_ids[0])
                ret = {
                    "embedding": embedding,
                    "token_num": token_num,
                }
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret



def release_worker_semaphore():
    worker.semaphore.release()

# 用limit_worker_concurrency实例化asyncio.Semaphore，然后调用其acquire方法

# semaphore管理一个内部计数器，该计数器在每次 acquire() 调用时递减，并在每次 release() 调用时递增。 
# 计数器永远不会低于零； 当 acquire() 发现它为零时，它会阻塞，等待其他线程调用release()。

# semaphore还支持上下文管理协议。可选参数给出内部计数器的初始值； 默认为 1。如果给定的值小于 0，则会引发 ValueError。
def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()

# BackgroundTasks为BackgroundTask的容器，通过add_task增加新的任务
# 该函数的意义是增加一个release_worker_semaphore任务
def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks

# asyncio.Semaphore,
# 调用worker的geenerate_strem_gate,
# 创建一个背景任务容器，在容器中增加一个release_worker_semaphore任务
# 调用starlette.response.StreamingResponse,StreamingResponse会
# 先执行流式输出，进行错误处理，然后执行背景任务,即调用semaphore.release方法。
@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)

# 与api_generate_stream类似，只是输出的时候不输出最后一条
@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}

@app.get("/",summary="swagger document")
async def docs():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        stream_interval=args.stream_interval,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
