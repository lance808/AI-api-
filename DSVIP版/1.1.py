import requests
import json
import os


class DeepSeekChatClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.history = []

        # 可配置参数
        self.model = "deepseek-chat"
        self.temperature = 0.8
        self.max_tokens = 2048
        self.stream = False
        self.keep_context = True
        self.api_version = "v1"  # 可切换v3-r1

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.stream else "application/json"
        }

    def chat(self, messages):
        """核心聊天方法"""
        params = {
            "model": self._get_model_name(),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream
        }

        try:
            response = requests.post(
                url=self.base_url,
                headers=self._get_headers(),
                json=params,
                stream=self.stream
            )
            response.raise_for_status()

            if self.stream:
                return self._handle_stream_response(response)
            else:
                return self._handle_normal_response(response)

        except requests.exceptions.RequestException as e:
            return f"请求失败: {str(e)}"

    def _get_model_name(self):
        """处理不同版本的模型名称"""
        model_mapping = {
            "v1": "deepseek-chat",
            "v3-r1": "deepseek-chat-32k"
        }
        return model_mapping.get(self.api_version, "deepseek-chat")

    def _handle_normal_response(self, response):
        """处理普通响应"""
        try:
            result = response.json()
            if 'choices' in result:
                content = result['choices'][0]['message']['content']
                if self.keep_context:
                    self._update_history(result['choices'][0]['message'])
                return content
            return "收到空响应"
        except json.JSONDecodeError:
            return "响应解析失败"

    def _handle_stream_response(self, response):
        """处理流式响应"""
        full_response = []
        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_data = json.loads(decoded_line[6:])
                        if 'choices' in json_data:
                            delta = json_data['choices'][0]['delta']
                            if 'content' in delta:
                                chunk = delta['content']
                                full_response.append(chunk)
                                yield chunk
            if self.keep_context and full_response:
                self._update_history({"role": "assistant", "content": "".join(full_response)})
        except Exception as e:
            yield f"\n流式传输中断: {str(e)}"

    def _update_history(self, message):
        """更新对话历史"""
        self.history.append(message)
        # 保持历史记录不超过5轮对话
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def interactive_chat(self):
        """交互式聊天界面"""
        print("DeepSeek 聊天客户端已启动（输入'exit'退出）")
        while True:
            user_input = input("\n您：")
            if user_input.lower() in ['exit', 'quit']:
                break

            # 构建消息列表
            messages = self.history.copy() if self.keep_context else []
            messages.append({"role": "user", "content": user_input})

            # 执行请求
            if self.stream:
                print("\n助手：", end="", flush=True)
                full_response = []
                for chunk in self.chat(messages):
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
                print()  # 换行
            else:
                response = self.chat(messages)
                print(f"\n助手：{response}")


def configure_client():
    """交互式配置客户端"""
    print("==== DeepSeek 客户端配置 ====")
    client = DeepSeekChatClient(os.getenv("DEEPSEEK_API_KEY"))

    # 流式模式
    stream = input("启用流式输出？(y/n): ").lower() == 'y'
    client.stream = stream

    # 上下文继承
    keep_context = input("保留对话上下文？(y/n): ").lower() == 'y'
    client.keep_context = keep_context

    # 模型版本
    model_version = input("选择模型版本 (1)v1 (2)v3-r1 [默认1]: ").strip()
    client.api_version = "v3-r1" if model_version == "2" else "v1"

    # 高级参数设置
    try:
        temp = float(input("设置temperature (0-2, 默认0.8): ") or 0.8)
        client.temperature = max(0, min(temp, 2))

        tokens = int(input("设置max_tokens (1-4096, 默认2048): ") or 2048)
        client.max_tokens = max(1, min(tokens, 4096))
    except ValueError:
        print("参数设置错误，使用默认值")

    return client


if __name__ == "__main__":
    # 安全获取API Key
    api_key = os.getenv("DEEPSEEK_API_KEY") or input("请输入API Key: ").strip()

    # 配置客户端
    client = configure_client()

    # 开始对话
    client.interactive_chat()