import requests
import json
from typing import Dict, Any, List, Optional


class DeepSeekClient:
    """DeepSeek API 客户端，用于与 DeepSeek 大语言模型进行交互"""

    def __init__(self,
                 api_key: str,
                 api_base: str = "https://api.deepseek.com/v1",
                 api_version: Optional[str] = None):
        """
        初始化 DeepSeek 客户端

        参数:
            api_key: DeepSeek API 密钥，必须通过参数提供
            api_base: API 基础 URL
            api_version: API 版本，可选
        """
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

        # 构建请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 添加 API 版本到请求头（如果提供）
        if self.api_version:
            self.headers["DeepSeek-Version"] = self.api_version

    def generate(self,
                 model: str,
                 messages: List[Dict[str, str]],
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 stream: bool = False) -> Dict[str, Any]:
        """
        调用 DeepSeek 模型生成文本

        参数:
            model: 要使用的模型名称，例如 "deepseek-chat"
            messages: 对话消息列表，每个消息是一个字典，包含 "role" 和 "content" 字段
            temperature: 温度参数，控制输出的随机性 (0.0 到 2.0 之间)
            max_tokens: 生成的最大 token 数
            stream: 是否使用流式响应

        返回:
            API 响应的 JSON 解析结果
        """
        endpoint = f"{self.api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()  # 检查请求是否成功

            if stream:
                # 处理流式响应
                return self._handle_stream_response(response)
            else:
                # 处理普通响应
                return response.json()

        except requests.exceptions.RequestException as e:
            # 处理请求异常
            print(f"API 请求失败: {e}")
            if response.status_code:
                print(f"状态码: {response.status_code}")
                print(f"错误详情: {response.text}")
            return {"error": str(e)}

    def _handle_stream_response(self, response: requests.Response) -> Dict[str, Any]:
        """处理流式 API 响应"""
        # 这里仅打印流式响应的每个部分
        # 实际应用中可以根据需要进行更复杂的处理
        print("流式响应内容:")
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            # 提取并处理 delta 内容
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            print(f"无法解析响应块: {data}")
        print()  # 换行
        return {"status": "stream_complete"}


# 使用示例 - 直接在代码中指定 API 信息
if __name__ == "__main__":
    # 初始化客户端并直接指定 API 密钥和基础 URL
    client = DeepSeekClient(
        api_key="sk-4b56587af95f4e9c8ae79089e48e7bc7",  # 必须替换为实际的 API 密钥
        api_base="https://api.deepseek.com/v1"  # 替换为实际的 API 基础 URL
    )

    # 定义对话消息
    messages = [
        {"role": "system", "content": "你是一个 helpful, creative, accurate, and harmless AI assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]

    # 调用模型生成回复
    response = client.generate(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        stream=False  # 设为 True 可以启用流式响应
    )

    # 打印回复
    if "error" in response:
        print(f"错误: {response['error']}")
    else:
        if not response.get("stream", False):
            # 非流式响应
            assistant_reply = response['choices'][0]['message']['content']
            print(f"助手回复: {assistant_reply}")