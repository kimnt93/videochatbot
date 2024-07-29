from celery import Celery
from langchain_core.runnables import Runnable
from celery.result import AsyncResult

app_celery = Celery('app')


@app_celery.task
def run_chain_task(chain: Runnable, run_input: dict):
    return chain.invoke(run_input)


def get_task_result(task_id):
    if task_id == "":
        return ""
    task = AsyncResult(task_id, app=app_celery)
    if task.state == 'SUCCESS':
        return task.result
    return ""  # incomplete
