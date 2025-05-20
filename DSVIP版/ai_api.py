import requests
import json
import time
from typing import Dict, Any, Optional, List, Union


class AIApiClient:
    """通用AI API调用客户端，支持API密钥认证"""

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


# 使用示例
if __name__ == "__main__":
    # 示例: 使用API密钥认证
    client = AIApiClient(
        base_url="https://api.example.com/v1",
        api_key="your_api_key_here",
        api_key_header="X-API-Key",  # 一些API使用不同的头名称
        api_key_prefix=""  # 某些API不需要前缀
    )

    # 调用API获取模型列表
    models = client.get("/models")
    print("可用模型:", models)

    # 调用需要认证的API端点
    response = client.post("/chat/completions", data={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, world!"}]
    })
    print("API响应:", response)