"""
Adapted from https://github.com/acheong08/Bard.
bard的本地api服务，即在本地起一个服务，通过post方法与谷歌的bard交互
"""
import argparse
import json
import random
import re
import string

from fastapi import FastAPI
import httpx
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import uvicorn


class ConversationState(BaseModel):
    conversation_id: str = ""
    response_id: str = ""
    choice_id: str = ""
    req_id: int = 0


class Message(BaseModel):
    content: str
    state: ConversationState = Field(default_factory=ConversationState)


class Response(BaseModel):
    content: str
    factualityQueries: Optional[List]
    textQuery: Optional[Union[str, List]]
    choices: List[dict]
    state: ConversationState


class Chatbot:
    """
    A class to interact with Google Bard.
    Parameters
        session_id: str
            The __Secure-1PSID cookie.
    """

    def __init__(self, session_id):
        headers = {
            "Host": "bard.google.com",
            "X-Same-Domain": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Origin": "https://bard.google.com",
            "Referer": "https://bard.google.com/",
        }
        # An asynchronous HTTP client, with connection pooling, HTTP/2, redirects,cookie persistence, etc.
        # 
        self.session = httpx.AsyncClient()
        self.session.headers = headers
        # Cookie，有时也用其复数形式 Cookies。类型为“小型文本文件”，是某些网站为了辨别用户身份，
        # 进行Session跟踪而储存在用户本地终端上的数据（通常经过加密），由用户客户端计算机暂时或永久保存的信息
        # 当客户机再次访问这个 Web 文档时这些信息可供该文档使用。由于“Cookie”具有可以保存在客户机上的神奇特性, 
        # 因此它可以帮助我们实现记录用户个人信息的功能, 而这一切都不必使用复杂的CGI等程序 [2] 。
        # 举例来说, 一个 Web 站点可能会为每一个访问者产生一个唯一的ID, 然后以 Cookie 文件的形式保存在每个用户的机器上。
        self.session.cookies.set("__Secure-1PSID", session_id)
        self.SNlM0e = None

    # 测试能否访问bard，如果能够访问，则返回文本中会包含SN1M0e:字段，匹配并返回
    async def _get_snlm0e(self):
        resp = await self.session.get(url="https://bard.google.com/", timeout=10)
        # Find "SNlM0e":"<ID>"
        if resp.status_code != 200:
            raise Exception("Could not get Google Bard")
        SNlM0e = re.search(r"SNlM0e\":\"(.*?)\"", resp.text).group(1)
        return SNlM0e
    # 
    async def ask(self, message: Message) -> Response:
        """
        Send a message to Google Bard and return the response.
        :param message: The message to send to Google Bard.
        :return: A dict containing the response from Google Bard.
        """
        if message.state.conversation_id == "":
            message.state.req_id = int("".join(random.choices(string.digits, k=4)))
        # url params
        params = {
            # "bl": "boq_assistant-bard-web-server_20230315.04_p2",
            # This is a newer API version
            "bl": "boq_assistant-bard-web-server_20230507.20_p2",
            "_reqid": str(message.state.req_id),
            "rt": "c",
        }

        # message arr -> data["f.req"]. Message is double json stringified
        message_struct = [
            [message.content],
            None,
            [
                message.state.conversation_id,
                message.state.response_id,
                message.state.choice_id,
            ],
        ]
        data = {
            "f.req": json.dumps([None, json.dumps(message_struct)]),
            "at": self.SNlM0e,
        }

        # do the request!
        # 先定义会话的id，然后定义url的参数，包括bl,_reqid,rt三个字段，_reqid即前序的会话id
        # 然后定义message_struct,将message_struct定义为data字典，字段依次为f.req,at
        resp = await self.session.post(
            "https://bard.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate",
            params=params,
            data=data,
            timeout=60,
        )

        chat_data = json.loads(resp.content.splitlines()[3])[0][2]
        if not chat_data:
            return Response(
                content=f"Google Bard encountered an error: {resp.content}.",
                factualityQueries=[],
                textQuery="",
                choices=[],
                state=message.state,
            )
        json_chat_data = json.loads(chat_data)
        conversation = ConversationState(
            conversation_id=json_chat_data[1][0],
            response_id=json_chat_data[1][1],
            choice_id=json_chat_data[4][0][0],
            req_id=message.state.req_id + 100000,
        )
        return Response(
            content=json_chat_data[0][0],
            factualityQueries=json_chat_data[3],
            textQuery=json_chat_data[2][0] if json_chat_data[2] is not None else "",
            choices=[{"id": i[0], "content": i[1]} for i in json_chat_data[4]],
            state=conversation,
        )


app = FastAPI()
chatbot = None

# on_event("startup"),定义在启动时的动作
# on_event("shutdown"),定义在结束时的动作
@app.on_event("startup")
async def startup_event():
    global chatbot
    # 从bard_cookie.json中读取cokie信息，将__Secure-1PSID的值作为ssession_id，
    # 以初始化Chatbot，测试能否访问bard，如果能够访问，则返回文本中会包含SN1M0e:字段，匹配并返回
    cookie = json.load(open("bard_cookie.json"))
    chatbot = Chatbot(cookie["__Secure-1PSID"])
    chatbot.SNlM0e = await chatbot._get_snlm0e()


@app.post("/chat", response_model=Response)
async def chat(message: Message):
    response = await chatbot.ask(message)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Google Bard worker")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=18900)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "bard_worker:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload,
    )
