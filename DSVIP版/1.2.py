import requests
import json
from typing import Dict, Any, List, Optional, Union, Generator


class DeepSeekClient:
    """DeepSeek API 客户端，用于与 DeepSeek 大语言模型进行交互"""

    # 内置支持的模型列表
    SUPPORTED_MODELS = {
        "deepseek-chat": ["v1", "v3-r1"],
        "deepseek-coder": ["v1", "v2"]
    }

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
        self.validate_api_key(api_key)
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.context = []  # 维护对话上下文

        # 构建请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 添加 API 版本到请求头（如果提供）
        if self.api_version:
            self.headers["DeepSeek-Version"] = self.api_version

    def validate_api_key(self, api_key: str) -> None:
        """验证API密钥格式"""
        if not api_key:
            raise ValueError("API密钥不能为空")
        if not api_key.startswith("sk-"):
            print("警告: API密钥格式可能不正确，应该以'sk-'开头")

    def generate(self,
                 model: str,
                 messages: List[Dict[str, str]],
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 stream: bool = False,
                 inherit_context: bool = True,
                 model_version: Optional[str] = None) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        调用 DeepSeek 模型生成文本

        参数:
            model: 要使用的模型名称，例如 "deepseek-chat"
            messages: 对话消息列表，每个消息是一个字典，包含 "role" 和 "content" 字段
            temperature: 温度参数，控制输出的随机性 (0.0 到 2.0 之间)
            max_tokens: 生成的最大 token 数
            stream: 是否使用流式响应
            inherit_context: 是否继承历史对话上下文
            model_version: 模型版本，如 "v1" 或 "v3-r1"

        返回:
            非流式响应: API 响应的 JSON 解析结果
            流式响应: 生成器，逐个返回文本块
        """
        # 检查模型版本
        full_model = self._get_full_model_name(model, model_version)

        # 构建请求消息（合并上下文）
        request_messages = self._build_request_messages(messages, inherit_context)

        endpoint = f"{self.api_base}/chat/completions"
        payload = {
            "model": full_model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()  # 检查请求是否成功

            if stream:
                # 处理流式响应
                return self._handle_stream_response(response, inherit_context)
            else:
                # 处理普通响应
                response_data = response.json()
                self._update_context(request_messages, response_data, inherit_context)
                return response_data

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                print("认证失败: 请检查您的API密钥是否正确")
            print(f"API 请求失败: {e}")
            if response.status_code:
                print(f"状态码: {response.status_code}")
                print(f"错误详情: {response.text}")
            return {"error": str(e)}
        except requests.exceptions.RequestException as e:
            print(f"网络请求异常: {e}")
            return {"error": str(e)}

    def _get_full_model_name(self, model: str, version: Optional[str]) -> str:
        """获取完整的模型名称（带版本）"""
        if not version:
            return model

        if model not in self.SUPPORTED_MODELS:
            print(f"警告: 模型 '{model}' 不在支持列表中")
            return f"{model}-{version}"

        if version not in self.SUPPORTED_MODELS[model]:
            print(f"警告: 模型 '{model}' 的版本 '{version}' 可能不受支持")

        return f"{model}-{version}"

    def _build_request_messages(self, messages: List[Dict[str, str]], inherit_context: bool) -> List[Dict[str, str]]:
        """构建请求消息（合并历史上下文）"""
        if not inherit_context:
            return messages

        # 合并历史上下文和当前消息
        return self.context + messages

    def _update_context(self, request_messages: List[Dict[str, str]], response_data: Dict[str, Any],
                        inherit_context: bool) -> None:
        """更新对话上下文"""
        if not inherit_context:
            return

        # 将用户消息添加到上下文
        user_messages = [msg for msg in request_messages if msg["role"] == "user"]
        self.context.extend(user_messages)

        # 将助手回复添加到上下文
        if "choices" in response_data and len(response_data["choices"]) > 0:
            assistant_message = response_data["choices"][0]["message"]
            self.context.append(assistant_message)

    def _handle_stream_response(self, response: requests.Response, inherit_context: bool) -> Generator[str, None, None]:
        """处理流式 API 响应，返回生成器并更新上下文"""
        full_content = ""

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_content += content
                                    yield content
                        except json.JSONDecodeError as e:
                            print(f"解析流式响应失败: {e}")
                            yield f"[解析错误: {e}]"

        # 更新上下文（如果需要）
        if inherit_context and full_content:
            self.context.append({"role": "assistant", "content": full_content})

        yield "[stream_end]"

    def clear_context(self) -> None:
        """清除对话上下文"""
        self.context = []

    def get_models(self) -> Dict[str, List[str]]:
        """获取支持的模型及其版本列表"""
        return self.SUPPORTED_MODELS


# 使用示例
if __name__ == "__main__":
    client = DeepSeekClient(
        api_key="sk-4b56587af95f4e9c8ae79089e48e7bc7",  # 必须替换为实际的 API 密钥
        api_base="https://api.deepseek.com/v1"
    )

    # 获取支持的模型
    print("支持的模型:", client.get_models())

    # 第一轮对话
    print("\n第一轮对话:")
    response = client.generate(
        model="deepseek-chat",
        model_version="v3-r1",
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}]
    )
    if "error" not in response:
        print("助手回复:", response['choices'][0]['message']['content'])

    # 第二轮对话（继承上下文）
    print("\n第二轮对话（继承上下文）:")
    response = client.generate(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "你能做什么？"}]
    )
    if "error" not in response:
        print("助手回复:", response['choices'][0]['message']['content'])

    # 第三轮对话（不继承上下文）
    print("\n第三轮对话（不继承上下文）:")
    response = client.generate(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "你是谁？"}]
    )
    if "error" not in response:
        print("助手回复:", response['choices'][0]['message']['content'])

    # 流式响应示例
    print("\n流式响应示例:")
    client.clear_context()  # 清除上下文
    stream_response = client.generate(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "讲一个简短的故事"}],
        stream=True
    )
    full_text = ""
    for chunk in stream_response:
        if chunk != "[stream_end]":
            full_text += chunk
            print(chunk, end='', flush=True)
    print("\n完整回复:", full_text)