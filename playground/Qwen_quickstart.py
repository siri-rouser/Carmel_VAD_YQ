import os
from openai import OpenAI

try:
    client = OpenAI(
        # 新加坡和北京地域的API Key不同。获取API Key：https://bailian.console.alibabacloud.com/?tab=model#/api-key
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 以下为新加坡地域url，若使用北京地域的模型，需将url替换为：https://dashscope.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",  
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
            ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://www.alibabacloud.com/help/zh/model-studio/developer-reference/error-code")