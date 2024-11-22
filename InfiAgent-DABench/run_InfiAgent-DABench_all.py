import asyncio
import logging
from functools import partial
from typing import List
import pandas as pd
from DABench import DABench
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import time

# 基础日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

async def process_item(key: str, da_bench: DABench, config_list: list, semaphore: asyncio.Semaphore):
    """处理单个评估项目"""
    async with semaphore:
        print(f"Processing {key}")
        start_time = time.time()  # 记录起始时间
        message = da_bench.get_prompt(key)
        assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
        user_proxy = UserProxyAgent(
            "user_proxy",
            code_execution_config={"work_dir": "coding", "use_docker": False},
            human_input_mode="NEVER"
        )

        # 将同步的initiate_chat包装成异步任务并添加超时
        try:
            # 使用 asyncio.get_event_loop().run_in_executor 来运行同步函数
            loop = asyncio.get_event_loop()
            rst = await asyncio.wait_for(
                loop.run_in_executor(None, partial(user_proxy.initiate_chat, assistant, message=message)),
                timeout=300
            )
        except asyncio.TimeoutError:
            logging.warning(f"Task {key} timed out after 300 seconds")
            return {
                'id': key,
                'prediction': "TIMEOUT",
                'label': str(da_bench.get_answer(key))
            }

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        prediction = rst.chat_history
        if prediction[-1]['content'] == 'TERMINATE' and prediction[-1]['name'] == 'assistant':
            prediction.pop()
            if prediction[-1]['content'] == '' and prediction[-1]['name'] == 'user_proxy':
                prediction.pop()

        prediction_str = "\n".join(f"Role: {p['role']}\nMessage: {p['content']}"
                                   for p in prediction)

        return {
            'id': key,
            'prediction': prediction_str,
            'label': str(da_bench.get_answer(key)),
            'time_elapsed': elapsed_time  # 添加耗时
        }

async def save_results(results: List[dict], filename: str = "output.xlsx"):
    """保存结果到Excel"""
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    logging.info(f"Results saved to {filename}")


async def main():
    """主评估函数"""
    DA = DABench()
    config_list = config_list_from_json(env_or_file="/home/manna/autogen/COST_CONFIG")
    # config_list = config_list_from_json(env_or_file="/home/manna/autogen/OAI_CONFIG_LIST_2")

    # 使用更大的并发限制
    semaphore = asyncio.Semaphore(50)

    # 准备任务列表
    items = list(DA.answers.items())
    tasks = []

    # keep_keys_list = [272, 207, 663, 651, 409, 665, 27, 733, 180, 657, 468, 363, 118, 666, 495, 522, 730, 408, 759, 550,
    #                   222, 530, 77, 480, 738, 35, 739, 721, 8, 734]
    # items = [item for item in items if item[0] in keep_keys_list]

    # 创建所有任务
    for key, _ in items:
        # if key not in exclude_id:
        task = asyncio.create_task(
            process_item(key, DA, config_list, semaphore)
        )
        tasks.append(task)

    # 使用 asyncio.gather 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 过滤出成功的结果
    valid_results = [r for r in results if isinstance(r, dict)]

    # 保存结果
    await save_results(valid_results, "autogen_deepseek_result.xlsx")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
