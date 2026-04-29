import asyncio


task_queue = asyncio.Queue()
# 定义一个异步函数，用于将用户输入的内容放入任务队列中
async def emit_task(content: str):
    # 将用户输入的内容放入任务队列中，供Agent工作协程处理
    await task_queue.put(content)