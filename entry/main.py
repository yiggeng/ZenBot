import os
import sys
import time
import asyncio
import random
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.styles import Style
from prompt_toolkit.application import get_app

from zenbot.core.multi_agent import create_multi_agent_app
from zenbot.core.config import DB_PATH
from zenbot.core.bus import task_queue
from zenbot.core.heartbeat import pacemaker_loop

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def type_line(text: str, delay: float = 0.008):
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print()

def print_banner():
    clear_screen()

    CYAN = '\033[38;5;51m'
    PURPLE = '\033[38;5;141m'
    SILVER = '\033[38;5;250m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    WHITE = '\033[37m'

    logo = f"""{CYAN}{BOLD}
 ███████╗███████╗███╗   ██╗██████╗  ██████╗ ████████╗
 ╚══███╔╝██╔════╝████╗  ██║██╔══██╗██╔═══██╗╚══██╔══╝
   ███╔╝ █████╗  ██╔██╗ ██║██████╔╝██║   ██║   ██║
  ███╔╝  ██╔══╝  ██║╚██╗██║██╔══██╗██║   ██║   ██║
 ███████╗███████╗██║ ╚████║██████╔╝╚██████╔╝   ██║
 ╚══════╝╚══════╝╚═╝  ╚═══╝╚═════╝  ╚═════╝    ╚═╝
{RESET}"""

    sub_title = f"{WHITE}{BOLD} [*] Welcome to the {PURPLE}{BOLD}ZenBot{RESET}{WHITE}{BOLD} !  {RESET}"

    quotes = [
        "It works on my machine.",
        "It compiles! Ship it.",
        "Git commit, push, pray.",
        "There's no place like 127.0.0.1.",
        "sudo make me a sandwich.",
        "Works fine in dev.",
        "May the source be with you.",
        "Ctrl+C, Ctrl+V, Deploy.",
        "Hello, World."
    ]
    quote = random.choice(quotes)
    meta = f" {SILVER}*{RESET} {CYAN}{quote}{RESET}"

    tip = (
        f"{PURPLE} * {RESET}"
        f"{SILVER}{PURPLE}{BOLD}ZenBot{RESET} 已完成启动。输入命令开始，输入 {PURPLE}/exit{RESET}{SILVER} 退出，{PURPLE}/new{RESET}{SILVER} 开启新会话。{RESET}\n"
    )

    print(logo)
    print(sub_title)
    print() 
    time.sleep(0.12)
    print(meta)
    print() 
    type_line(tip, delay=0.004)


def cprint(text="", end="\n"):
    print_formatted_text(ANSI(str(text)), end=end)


async def async_main():
    print_banner()
    
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)
    
    current_provider = os.getenv("DEFAULT_PROVIDER", "aliyun")
    current_model = os.getenv("DEFAULT_MODEL", "glm-5")

    #创建 Multi-Agent app，并连接到异步SQLite检查点存储
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        app = create_multi_agent_app(provider_name=current_provider, model_name=current_model, checkpointer=memory)
        config = {"configurable": {"thread_id": "zenbot_main"}}
        session_counter = [0]
        current_stream_task = [None]  # 当前正在跑的 astream task，供 Ctrl+C 取消

        class SpinnerState:
            action_words = [
                "Thinking...",              
                "Working...",               
                "Beep boop...",             
                "Eating bugs...",           
                "Charging battery...",      
                "Brewing coffee...",        
                "Blinking lights...",       
                "Polishing pixels...",      
                "Scanning matrix...",       
                "Warming up circuits...",   
                "Syncing data...",          
                "Pinging server..."         
            ]
            current_words = [] 
            is_spinning = False
            start_time = 0
            frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            is_tool_calling = False
            tool_msg = ""
            awaiting_confirm = False  # 是否正在等待用户确认 Planner 方案

        spinner = SpinnerState()

        # 定义底部工具栏显示函数，根据旋转器状态动态更新显示内容
        def get_custom_toolbar():
            if not spinner.is_spinning:
                return ANSI("") 
            # 计算旋转动画帧和提示词索引，构建显示字符串
            elapsed = time.time() - spinner.start_time
            # 如果正在调用工具，显示工具调用信息；否则显示旋转提示词
            if spinner.is_tool_calling:
                display_msg = spinner.tool_msg
            else:
                idx_word = int(elapsed) % len(spinner.current_words)
                display_msg = f"* {spinner.current_words[idx_word]}"

            idx_frame = int(elapsed * 12) % len(spinner.frames)
            frame = spinner.frames[idx_frame]
            

            return ANSI(f"  \033[38;5;51m{frame}\033[0m \033[38;5;250m{display_msg}\033[0m \033[38;5;141m[{elapsed:.1f}s]\033[0m")

        prompt_message = ANSI("  \033[38;5;51m>\033[0m ")
        placeholder_text = ANSI("\033[3m\033[38;5;242minput...\033[0m")

        # Agent核心工作协程：负责处理用户输入，驱动Agent决策，并更新状态
        async def agent_worker():
            while True:
                user_input = await task_queue.get()
                if user_input.lower() in ["/exit", "/quit"]:
                    task_queue.task_done()
                    break

                if user_input.lower() == "/new":
                    session_counter[0] += 1
                    new_thread_id = f"zenbot_main_{session_counter[0]}"
                    config["configurable"]["thread_id"] = new_thread_id
                    cprint(f"  \033[38;5;141m* 新会话已开启 (#{session_counter[0]})，历史记忆已隔离。\033[0m\n")
                    task_queue.task_done()
                    continue

                spinner.current_words = spinner.action_words.copy()
                random.shuffle(spinner.current_words)
                spinner.start_time = time.time()
                spinner.is_spinning = True
                spinner.is_tool_calling = False

                inputs = {
                    "user_input": user_input, "messages": [],
                    "summary": "", "final_answer": ""
                }

                async def _handle_interrupt(iv: dict):
                    itype = iv.get("type", "plan_approval")
                    if itype == "plan_approval":
                        plan_text = iv.get("plan", "")
                        task_count = iv.get("task_count", 0)
                        mode_label = iv.get("mode", "并行")
                        label = "重新拆解出" if iv.get("replan") else "拆解出"
                        cprint(f"\n  \033[38;5;51m✦ Planner {label} {task_count} 个{mode_label}子任务：\033[0m")
                        cprint(f"\033[38;5;242m{plan_text}\033[0m")
                        cprint(f"\n  \033[38;5;250m确认执行？(y/n)\033[0m ")
                    elif itype == "plan_feedback":
                        cprint(f"\n  \033[38;5;214m✦ {iv.get('prompt', '请输入改进建议：')}\033[0m ")
                    spinner.is_spinning = False
                    spinner.awaiting_confirm = True
                    user_reply = await task_queue.get()
                    task_queue.task_done()
                    spinner.start_time = time.time()
                    spinner.is_spinning = True
                    spinner.is_tool_calling = False
                    return user_reply

                def _handle_node(rnode, rdata):
                    if rdata is None:
                        return
                    if rnode == "planner":
                        spinner.tool_msg = "Planner 拆解任务中..."
                    elif rnode == "worker":
                        spinner.tool_msg = "Workers 执行中..."
                    elif rnode == "aggregator":
                        spinner.is_spinning = False
                        answer = rdata.get("final_answer", "")
                        if answer and answer != "__replan__":
                            lines = answer.strip().split('\n')
                            formatted_out = f"  \033[38;5;141m>\033[0m \033[38;5;250m{lines[0]}"
                            for line in lines[1:]:
                                formatted_out += f"\n    {line}"
                            formatted_out += "\033[0m"
                            cprint(formatted_out)

                next_input = inputs

                async def _run_stream():
                    nonlocal next_input
                    while True:
                        got_interrupt = False
                        async for event in app.astream(next_input, config, stream_mode="updates"):
                            for node_name, node_data in event.items():
                                if node_name == "__interrupt__":
                                    got_interrupt = True
                                    i_item = node_data[0] if node_data else None
                                    i_val = getattr(i_item, "value", None) or {}
                                    if not isinstance(i_val, dict):
                                        i_val = {}
                                    user_reply = await _handle_interrupt(i_val)
                                    next_input = Command(resume=user_reply)
                                else:
                                    _handle_node(node_name, node_data)
                        if not got_interrupt:
                            break

                stream_task = asyncio.create_task(_run_stream())
                current_stream_task[0] = stream_task
                try:
                    await stream_task
                except asyncio.CancelledError:
                    spinner.is_spinning = False
                    cprint(f"\n  \033[38;5;214m[ ⏹ 已中断，可继续输入 ]\033[0m")
                except Exception as e:
                    spinner.is_spinning = False
                    cprint(f"  \033[31m[ ⚠️ 引擎异常 : {e} ]\033[0m")
                finally:
                    current_stream_task[0] = None

                spinner.is_spinning = False
                cprint()
                task_queue.task_done()

        async def user_input_loop():
            custom_style = Style.from_dict({
                'bottom-toolbar': 'bg:default fg:default noreverse',
            })

            session = PromptSession(
                bottom_toolbar=get_custom_toolbar,
                style=custom_style,
                erase_when_done=True,
                reserve_space_for_menu=0  
            )
            
            async def redraw_timer():
                while True:
                    if spinner.is_spinning:
                        try:
                            get_app().invalidate()
                        except Exception:
                            pass
                    await asyncio.sleep(0.08)
                    
            redraw_task = asyncio.create_task(redraw_timer())
            
            while True:
                try:
                    user_input = await session.prompt_async(prompt_message, placeholder=placeholder_text)

                    user_input = user_input.strip()
                    if not user_input:
                        continue

                    # 等待确认时不显示气泡，直接入队
                    if spinner.awaiting_confirm:
                        spinner.awaiting_confirm = False
                        await task_queue.put(user_input)
                        continue

                    padded_bubble = f"  > {user_input}    "
                    cprint(f"\033[48;2;38;38;38m\033[38;5;255m{padded_bubble}\033[0m\n")
                    
                    await task_queue.put(user_input)
                    if user_input.lower() in ["/exit", "/quit"]:
                        cprint("  \033[38;5;141m* 记忆已固化，ZenBot 进入休眠。\033[0m")
                        break
                        
                except (KeyboardInterrupt, EOFError):
                    task = current_stream_task[0]
                    if task and not task.done():
                        task.cancel()
                        cprint("\n  \033[38;5;214m[ ⏹ 操作已中断，可继续输入 ]\033[0m")
                    else:
                        cprint("\n  \033[38;5;141m* 强制中断，ZenBot 进入休眠。\033[0m")
                        await task_queue.put("/exit")
                        break

            redraw_task.cancel() 
        # 使用patch_stdout确保在异步环境中正确处理标准输出，启动Agent工作协程和心跳协程，同时运行用户输入循环，等待任务完成后清理协程资源
        with patch_stdout():
            worker = asyncio.create_task(agent_worker())
            heartbeat_worker = asyncio.create_task(pacemaker_loop(check_interval=10))
            await user_input_loop()
            await task_queue.join()
            worker.cancel()
            heartbeat_worker.cancel()

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()