import requests
import json
import time
from typing import Dict, Any, Optional, List, Union, Callable


class AIApiClient:
    """通用AI API调用客户端，支持API密钥认证和对话管理"""

    def __init__(
            self,
            base_url: str,
            api_key: str,
            api_key_header: str = "Authorization",
            api_key_prefix: str = "Bearer",
            timeout: int = 60,
            max_retries: int = 3,
            retry_delay: int = 5
    ):
        """
        初始化API客户端

        参数:
            base_url: API的基础URL
            api_key: API密钥
            api_key_header: API密钥在请求头中的名称
            api_key_prefix: API密钥的前缀(例如"Bearer")
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            retry_delay: 重试间隔时间(秒)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.api_key_prefix = api_key_prefix
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.conversation_history = []

        print(f"初始化API客户端: {base_url}")

    def _prepare_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """准备请求头，添加API密钥"""
        final_headers = headers or {}

        # 添加内容类型(如果未提供)
        if "Content-Type" not in final_headers:
            final_headers["Content-Type"] = "application/json"

        # 添加API密钥
        prefix = f"{self.api_key_prefix} " if self.api_key_prefix else ""
        final_headers[self.api_key_header] = f"{prefix}{self.api_key}"

        return final_headers

    def _make_request(
            self,
            method: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            data: Optional[Union[Dict[str, Any], str]] = None,
            headers: Optional[Dict[str, str]] = None,
            files: Optional[Dict[str, Any]] = None,
            stream: bool = False
    ) -> Dict[str, Any]:
        """
        发送API请求，处理重试逻辑

        参数:
            method: HTTP方法(GET, POST, PUT, DELETE等)
            endpoint: API端点路径
            params: 查询参数
            data: 请求体数据
            headers: 请求头
            files: 文件上传
            stream: 是否流式响应

        返回:
            API响应的JSON数据
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._prepare_headers(headers)

        # 处理请求体数据
        if data and headers.get("Content-Type") == "application/json" and not files:
            if isinstance(data, dict):
                data = json.dumps(data)

        # 实现重试逻辑
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    files=files,
                    timeout=self.timeout,
                    stream=stream
                )

                # 检查HTTP状态码
                if 200 <= response.status_code < 300:
                    if stream:
                        return response
                    else:
                        try:
                            return response.json()
                        except json.JSONDecodeError:
                            return {"text": response.text}
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # 这些状态码通常表示临时错误，可以重试
                    print(f"请求失败，状态码: {response.status_code}，尝试重试 ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                    continue
                else:
                    # 其他状态码表示客户端错误或不可恢复的服务器错误
                    raise Exception(f"API请求失败: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"请求异常: {e}，尝试重试 ({attempt + 1}/{self.max_retries})")
                time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避

        # 所有重试都失败
        raise Exception("达到最大重试次数后请求仍然失败")

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """发送GET请求"""
        return self._make_request("GET", endpoint, params=params, **kwargs)

    def post(self, endpoint: str, data: Optional[Union[Dict[str, Any], str]] = None, **kwargs) -> Dict[str, Any]:
        """发送POST请求"""
        return self._make_request("POST", endpoint, data=data, **kwargs)

    def put(self, endpoint: str, data: Optional[Union[Dict[str, Any], str]] = None, **kwargs) -> Dict[str, Any]:
        """发送PUT请求"""
        return self._make_request("PUT", endpoint, data=data, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送DELETE请求"""
        return self._make_request("DELETE", endpoint, **kwargs)

    def start_conversation(self) -> None:
        """开始一个新的对话"""
        self.conversation_history = []
        print("开始新的对话")

    def send_message(self, endpoint: str, message: str, model: str = "gpt-4") -> Dict[str, Any]:
        """
        发送消息到AI API并获取回复

        参数:
            endpoint: API端点路径
            message: 用户消息
            model: 使用的AI模型

        返回:
            API响应
        """
        # 添加用户消息到对话历史
        self.conversation_history.append({"role": "user", "content": message})

        # 准备请求数据
        data = {
            "model": model,
            "messages": self.conversation_history
        }

        # 发送请求
        response = self.post(endpoint, data=data)

        # 提取回复并添加到对话历史
        if "choices" in response and len(response["choices"]) > 0:
            assistant_reply = response["choices"][0]["message"]["content"]
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})

        return response

    def run_conversation(
            self,
            endpoint: str,
            model: str = "gpt-4",
            max_turns: int = 0,
            custom_response_processor: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> None:
        """
        运行对话

        参数:
            endpoint: API端点路径
            model: 使用的AI模型
            max_turns: 最大对话轮数(0表示无限制，手动控制)
            custom_response_processor: 自定义响应处理器
        """
        self.start_conversation()
        turn_count = 0

        while True:
            # 获取用户输入
            user_input = input("\n你: ")

            # 检查是否退出
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("对话已结束")
                break

            # 发送消息
            try:
                response = self.send_message(endpoint, user_input, model)

                # 处理响应
                if custom_response_processor:
                    custom_response_processor(response)
                else:
                    if "choices" in response and len(response["choices"]) > 0:
                        assistant_reply = response["choices"][0]["message"]["content"]
                        print(f"\nAI: {assistant_reply}")
                    else:
                        print("\nAI: [无回复]")

            except Exception as e:
                print(f"错误: {e}")

            # 更新对话轮数
            turn_count += 1

            # 检查是否达到最大轮数
            if max_turns > 0 and turn_count >= max_turns:
                print(f"\n已达到最大对话轮数({max_turns})")
                break


# 使用示例
if __name__ == "__main__":
    # 初始化API客户端
    client = AIApiClient(
        base_url="https://api.example.com/v1",
        api_key="your_api_key_here"
    )

    # 示例1: 固定次数的对话
    print("\n=== 固定次数的对话 (3轮) ===")
    client.run_conversation(
        endpoint="/chat/completions",
        model="gpt-4",
        max_turns=3
    )

    # 示例2: 手动控制的对话
    print("\n=== 手动控制的对话 (输入'exit'结束) ===")
    client.run_conversation(
        endpoint="/chat/completions",
        model="gpt-4"
    )

    # 示例3: 带自定义响应处理器的对话
    print("\n=== 带自定义响应处理器的对话 ===")


    def custom_processor(response: Dict[str, Any]) -> None:
        if "choices" in response and len(response["choices"]) > 0:
            reply = response["choices"][0]["message"]["content"]
            print(f"\n自定义处理器 - AI回复: {reply[:50]}... (截断显示)")


    client.run_conversation(
        endpoint="/chat/completions",
        model="gpt-4",
        custom_response_processor=custom_processor
    )