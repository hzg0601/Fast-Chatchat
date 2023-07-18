"""
A controller manages distributed workers.
It sends worker addresses to clients.
定义Controller类，该类包含如下主要方法：
1. register_worker: 
    调用get_worker_status执行worker的访问，如果调用失败返回False,否则返回True;get_worker_status执行requests.post方法访问"worker_get_status"方法;
2. get_worker_address:
    根据dispatch_method和model_name找到对应的worker_name
3.  worker_api_get_status: 
    针对api的get_worker_stutus函数
4. worker_api_generate_stream：
    首先调用get_worker_address获取worker_name,然拼接{worker_name}+/worker_generate_stream得到流式输出的接口地址,
    执行requests.post方法访问该接口地址，获取流式输出的生成器，执行流式输出。
    从此处可以看出，后续仍需针对每个worker定义{worker_name}+/worker_generate_stream接口的函数。
"""
import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import time
from typing import List, Union
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from fastchat.constants import (
    CONTROLLER_HEART_BEAT_EXPIRATION,
    WORKER_API_TIMEOUT,
    ErrorCode,
    SERVER_ERROR_MSG,
)
from fastchat.utils import build_logger


logger = build_logger("controller", "controller.log")

#? auto实例就是一个object实例，怎么实现分派的？ 
class DispatchMethod(Enum):
    # 在Enum中auto()会自动将实例转换为对应的值
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")

#? 这些数据是如何给定的？
@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str

# 先等待CONTROLLER_HEART_BEAT_EXPIRATION秒
# 然后执行controller的remove_stable_workers_by_expiration方法
# 删除worker_info里对应worker_name的信息
def heart_beat_controller(controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        # 返回DispatchMethod.LOTTERY或DispatchMethod.SHORTEST_QUEUE
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        # 构造一个Thread类，在执行时调用heart_beat_controller,
        # 即先等待一段时间，然后执行remove_stable_workers_by_expiration
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,)
        )
        # Start the thread's activity.
        self.heart_beat_thread.start()
    # 调用get_worker_status执行worker的访问，如果调用失败返回False,否则返回True
    # get_worker_status执行requests.post方法访问"worker_get_status"方法，
    # 访问成功则得到状态信息，否则返回None.
    def register_worker(
        self, worker_name: str, check_heart_beat: bool, worker_status: dict
    ):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"],
            worker_status["speed"],
            worker_status["queue_length"],
            check_heart_beat,
            time.time(),
        )

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True
    # get_worker_status执行requests.post方法访问{worker_name}+/worker_get_status
    # 访问成功则得到状态信息，否则返回None.
    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()
    #? remove_worker仅仅是删除了worker_info里对应worker_name的信息
    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]
    # 如果调用register_worker即访问worker_name/worker_get_status失败，则
    # 认为该worker已失效，日志记录该信息
    #? refresh_all_workers也仅仅是测试对应worker的worker_get_status是否能访问成功，并没有更新信息？
    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)
    
    # 根据dispatch_method和model_name找到对应的worker_name

    def get_worker_address(self, model_name: str):
        # 如果dispatch_method的模式为LOTTERY
        # 则记录所有worker的speed字段的值，组装为一个np.array并求和
        # 如果速度和<1e-4,则返回"";否则根据速度和对速度进行标准化，按速度随机采样一个worker_name并返回
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            #? 这种写法意义在哪里？
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            while True:
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        # 如果dispatch_method的模式为SHORTEST_QUEUE，
        # 则记录所有worker的worker_name,以及用speed标准化的queue_length
        # 如果worker_names的数量不为0，则找到最小的的标准化queue_length的index
        # 找到对应的worker_name,将它的queue_length+1,然后返回worker_name.
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            #? 为什么将queue_length+1
            self.worker_info[w_name].queue_length += 1
            logger.info(
                f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}"
            )
            return w_name
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")
    # 将给定worker_name的last_heart_beat设定为当前时间，queue_length设定为给定的queue_length
    def receive_heart_beat(self, worker_name: str, queue_length: int):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True
    #? 笔误吧，应为remove_stale_workers_by_expiration
    # 如果某个worker的last_heart_beat时间超出了CONTROLLER_HEART_BEAT_EXPIRATION，且check_heart_beat字段的值为True
    # 则执行remove_worker，删除worker_info里对应worker_name的信息
    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)
    # 没有指定worker的处理逻辑，针对api
    def handle_no_worker(params):
        logger.info(f"no worker: {params['model']}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": ErrorCode.CONTROLLER_NO_WORKER,
        }
        return json.dumps(ret).encode() + b"\0"
    # 访问worker_address失败的处理逻辑,针对api
    def handle_worker_timeout(worker_address):
        logger.info(f"worker timeout: {worker_address}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": ErrorCode.CONTROLLER_WORKER_TIMEOUT,
        }
        return json.dumps(ret).encode() + b"\0"

    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    # 针对api的get_worker_status处理
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }
    # 首先根据model_name获取model_name，然拼接{worker_name}+/worker_generate_stream得到流式输出的接口地址
    # 
    def worker_api_generate_stream(self, params):
        # get_worker_addgress根据model_name获取worker_name
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            yield self.handle_no_worker(params)

        try:
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                json=params,
                stream=True,
                timeout=WORKER_API_TIMEOUT,
            )
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            yield self.handle_worker_timeout(worker_addr)


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_name"], data["check_heart_beat"], data.get("worker_status", None)
    )


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    params = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return controller.worker_api_get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
