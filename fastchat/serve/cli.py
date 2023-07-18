"""
Chat with a model with command line interface.
定义命令行输出的形式类：SimpleChatIO,RichChatIO,ProgrammaticChatIO,
基于chat_loop提供命令行客户端的命令行对话
Usage:
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0

Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
"""
import argparse
import os
import re
import sys
# prompt_toolkit is a Library for building powerful interactive
# command lines in Python. It can be a replacement for GNU Readline, but it can be much more than that.

# PromptSession 用于提示应用程序，可以用作 GNU Readline 的替代品。 这是许多prompt_toolkit功能的包装，可以替代raw_input。 
# 所有需要“格式化文本”的参数都可以采用纯文本（unicode 对象）、(style_str, text) 元组列表或 HTML 对象。
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
# rich: Rich text and beautiful formatting in the terminal.
# Console: A high level console interface.
from rich.console import Console
#Live: Renders an auto-updating live display of any given renderable.
from rich.live import Live
# Markdown: A Markdown renderable.
from rich.markdown import Markdown

# add_model_args 增加命令行选项的函数
from fastchat.model.model_adapter import add_model_args
# GptqConfig: gptq参数的dataclass
from fastchat.modules.gptq import GptqConfig
# ChatIO：组织输入、输出和流式输出的抽象类
# chat_loop:从加载模型、模板、stream_function到得到输出的全流程函数
# 它的输出、输出、流式输出的组织是基于ChatIO
from fastchat.serve.inference import ChatIO, chat_loop

# 一个组织输入、输出、stream_output模板的基本类
class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

# 在终端以rich支持的富格式输出
class RichChatIO(ChatIO):
    bindings = KeyBindings()
    # Decorator for adding a key bindings.add(keys,filter,eager,is_global,save_before,record_in_macro)
    # 即在会话中，按下escape键，enter键相当于执行了该函数，在本例中就是event.app.current_buffer.newline()
    @bindings.add("escape", "enter")
    def _(event):
        event.app.current_buffer.newline()

    def __init__(self, multiline: bool = False, mouse: bool = False):
        # 定义prompt_toolkit命令行prompt会话类实例
        self._prompt_session = PromptSession(history=InMemoryHistory())
        # 定义prompt_toolkit自动补全的实例
        self._completer = WordCompleter(
            words=["!!exit", "!!reset"], pattern=re.compile("$")
        )
        # 使用rich的Console作为输出的默认console
        self._console = Console()
        self._multiline = multiline
        self._mouse = mouse

    def prompt_for_input(self, role) -> str:
        # 格式 print(f"[red][bold][itatic][underline][blink2]role:")
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        # The first set of arguments is a subset of the ~.PromptSession class itself. 
        # For these, passing in None will keep the current values that are active in the session. 
        # Passing in a value will set the attribute for the session,
        #  which means that it applies to the current, but also to the next prompts.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            mouse_support=self._mouse,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings if self._multiline else None,
        )
        print("*"*80)
        print("test _prompt_session.prompt")
        print(prompt_input)
        self._console.print()
        print("*"*80)
        print("test _console.print()")
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        # todo above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                print("*"*80,"test markdown..")
                live.update(markdown)
        self._console.print()
        return text


class ProgrammaticChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        contents = ""
        # `end_sequence` signals the end of a message. It is unlikely to occur in
        #  message content.
        end_sequence = " __END_OF_A_MESSAGE_47582648__\n"
        len_end = len(end_sequence)
        # 如果接收的数据不足end_sequence的长度，则持续接收数据，否则判断最后的字符是否是end_sequence
        # 如果是就停止读取数据返回content,格式化打印[!OP:{role}: {content}]
        while True:
            if len(contents) >= len_end:
                last_chars = contents[-len_end:]
                if last_chars == end_sequence:
                    break
            try:
                char = sys.stdin.read(1)
                contents = contents + char
            except EOFError:
                continue
        contents = contents[:-len_end]
        print(f"[!OP:{role}]: {contents}", flush=True)
        return contents

    def prompt_for_output(self, role: str):
        print(f"[!OP:{role}]: ", end="", flush=True)

    def stream_output(self, output_stream):
        # 读取输出流generator的文本，根据空格划分，定义now为文本长度-1，
        # 若now大于前序文本的长度，则打印[pre:now]间的文本
        # 打印最后一次输出的pre后的所有文本，返回最后一次的全部文本
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            #* ---应只适用于英文，不适用于中文-----
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        chat_loop(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.temperature,
            args.repetition_penalty,
            args.max_new_tokens,
            chatio,
            GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            ),
            args.revision,
            args.judge_sent_end,
            args.debug,
            history=not args.no_history,
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
