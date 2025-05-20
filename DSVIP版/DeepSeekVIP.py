import sys
import json
import time
import requests
from typing import Dict, Any, Optional, List, Union
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QLabel, QComboBox, QSpinBox, QMessageBox, QTabWidget,QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor


class ApiRequestThread(QThread):
    """API请求线程，用于在后台执行网络请求"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, client, endpoint, data):
        super().__init__()
        self.client = client
        self.endpoint = endpoint
        self.data = data

    def run(self):
        try:
            response = self.client.post(self.endpoint, data=self.data)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))


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
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.api_key_prefix = api_key_prefix
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _prepare_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        final_headers = headers or {}

        if "Content-Type" not in final_headers:
            final_headers["Content-Type"] = "application/json"

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
        url = f"{self.base_url}{endpoint}"
        headers = self._prepare_headers(headers)

        if data and headers.get("Content-Type") == "application/json" and not files:
            if isinstance(data, dict):
                data = json.dumps(data)

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

                if 200 <= response.status_code < 300:
                    if stream:
                        return response
                    else:
                        try:
                            return response.json()
                        except json.JSONDecodeError:
                            return {"text": response.text}
                elif response.status_code in [429, 500, 502, 503, 504]:
                    print(f"请求失败，状态码: {response.status_code}，尝试重试 ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise Exception(f"API请求失败: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"请求异常: {e}，尝试重试 ({attempt + 1}/{self.max_retries})")
                time.sleep(self.retry_delay * (2 ** attempt))

        raise Exception("达到最大重试次数后请求仍然失败")

    def post(self, endpoint: str, data: Optional[Union[Dict[str, Any], str]] = None, **kwargs) -> Dict[str, Any]:
        return self._make_request("POST", endpoint, data=data, **kwargs)

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """发送GET请求"""
        return self._make_request("GET", endpoint, params=params, **kwargs)


class Conversation:
    """对话管理类"""

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.history = []
        self.turn_count = 0
        self.max_turns = 0  # 0表示无限制

    def start_new(self, model: Optional[str] = None) -> None:
        """开始新对话"""
        if model:
            self.model = model
        self.history = []
        self.turn_count = 0

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        """添加AI回复"""
        self.history.append({"role": "assistant", "content": message})
        self.turn_count += 1

    def get_messages(self) -> List[Dict[str, str]]:
        """获取所有消息"""
        return self.history

    def can_continue(self) -> bool:
        """检查是否可以继续对话"""
        return self.max_turns == 0 or self.turn_count < self.max_turns

    def set_max_turns(self, max_turns: int) -> None:
        """设置最大对话轮数"""
        self.max_turns = max_turns


class AIChatApp(QMainWindow):
    """AI聊天应用主窗口"""

    def __init__(self):
        super().__init__()

        self.client = None
        self.conversation = Conversation()

        self.init_ui()

    def init_ui(self) -> None:
        """初始化用户界面"""
        self.setWindowTitle("DeepSeek AI聊天助手")
        self.setGeometry(100, 100, 800, 600)

        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 创建标签页
        self.tabs = QTabWidget()

        # API设置标签页
        self.setup_tab = self.create_setup_tab()
        self.tabs.addTab(self.setup_tab, "API设置")

        # 聊天标签页
        self.chat_tab = self.create_chat_tab()
        self.tabs.addTab(self.chat_tab, "聊天")

        # 添加标签页到主布局
        main_layout.addWidget(self.tabs)

        # 设置中央部件
        self.setCentralWidget(main_widget)

    def create_setup_tab(self) -> QWidget:
        """创建API设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # API基础URL - 预设DeepSeek API
        url_layout = QHBoxLayout()
        url_label = QLabel("API基础URL:")
        self.url_input = QLineEdit("https://api.deepseek.com/v1")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)

        # API密钥
        key_layout = QHBoxLayout()
        key_label = QLabel("API密钥:")
        self.key_input = QLineEdit("")#测试用记得删除
        self.key_input.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.key_input)

        # 显示密码复选框
        self.show_password_checkbox = QCheckBox("显示密码")
        self.show_password_checkbox.toggled.connect(self.toggle_password_visibility)
        key_layout.addWidget(self.show_password_checkbox)

        layout.addLayout(key_layout)

        # API密钥头部 - 预设DeepSeek格式
        header_layout = QHBoxLayout()
        header_label = QLabel("密钥头部:")
        self.header_input = QLineEdit("Authorization")
        header_layout.addWidget(header_label)
        header_layout.addWidget(self.header_input)
        layout.addLayout(header_layout)

        # API密钥前缀 - 预设DeepSeek格式
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel("密钥前缀:")
        self.prefix_input = QLineEdit("Bearer")
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_input)
        layout.addLayout(prefix_layout)

        # 测试连接按钮
        self.test_button = QPushButton("测试连接")
        self.test_button.clicked.connect(self.test_api_connection)
        layout.addWidget(self.test_button)

        # 状态标签
        self.status_label = QLabel("未连接")
        layout.addWidget(self.status_label)

        # 垂直间隔
        layout.addStretch()

        return tab

    def toggle_password_visibility(self, checked):
        """切换API密钥的可见性"""
        if checked:
            self.key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.key_input.setEchoMode(QLineEdit.Password)

    def create_chat_tab(self) -> QWidget:
        """创建聊天标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 聊天设置区域
        settings_layout = QHBoxLayout()

        # 模型选择 - 预设DeepSeek模型
        model_label = QLabel("模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-chat", "deepseek-coder", "deepseek-vision"])
        settings_layout.addWidget(model_label)
        settings_layout.addWidget(self.model_combo)

        # 最大对话轮数
        turns_label = QLabel("最大对话轮数:")
        self.turns_spin = QSpinBox()
        self.turns_spin.setRange(0, 100)
        self.turns_spin.setValue(0)
        self.turns_spin.setSuffix(" (0=无限制)")
        settings_layout.addWidget(turns_label)
        settings_layout.addWidget(self.turns_spin)

        # 新对话按钮
        self.new_chat_button = QPushButton("开始新对话")
        self.new_chat_button.clicked.connect(self.start_new_chat)
        settings_layout.addWidget(self.new_chat_button)

        layout.addLayout(settings_layout)

        # 聊天历史显示区域
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setAcceptRichText(True)
        layout.addWidget(self.chat_history)

        # 输入区域
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("输入消息...")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)

        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        return tab

    def test_api_connection(self) -> None:
        """测试API连接"""
        try:
            # 创建API客户端
            self.client = AIApiClient(
                base_url=self.url_input.text(),
                api_key=self.key_input.text(),
                api_key_header=self.header_input.text(),
                api_key_prefix=self.prefix_input.text()
            )

            # 测试请求
            self.status_label.setText("测试中...")
            response = self.client.get("/models")

            if "models" in str(response) or "data" in response:
                self.status_label.setText("连接成功")
                QMessageBox.information(self, "成功", "API连接测试成功!")
            else:
                self.status_label.setText("连接失败")
                QMessageBox.warning(self, "失败", f"API返回意外响应: {response}")

        except Exception as e:
            self.status_label.setText("连接失败")
            QMessageBox.critical(self, "错误", f"连接测试失败: {str(e)}")

    def start_new_chat(self) -> None:
        """开始新对话"""
        self.conversation.start_new(self.model_combo.currentText())
        self.conversation.set_max_turns(self.turns_spin.value())
        self.chat_history.clear()
        self.chat_history.append("<b>=== 新对话开始 ===</b>")

    def send_message(self) -> None:
        """发送消息到AI API"""
        # 检查是否已连接API
        if not self.client:
            QMessageBox.warning(self, "未连接", "请先在API设置中测试连接")
            return

        # 获取用户输入
        message = self.message_input.text().strip()
        if not message:
            return

        # 检查是否可以继续对话
        if not self.conversation.can_continue():
            QMessageBox.information(self, "对话结束", "已达到最大对话轮数")
            return

        # 添加用户消息到历史
        self.conversation.add_user_message(message)
        self.chat_history.append(f"<b>你:</b> {message}")
        self.message_input.clear()

        # 准备请求数据
        data = {
            "model": self.conversation.model,
            "messages": self.conversation.get_messages()
        }

        # 在后台线程中发送请求
        self.send_button.setEnabled(False)
        self.message_input.setReadOnly(True)

        self.thread = ApiRequestThread(self.client, "/chat/completions", data)
        self.thread.finished.connect(self.handle_api_response)
        self.thread.error.connect(self.handle_api_error)
        self.thread.start()

    def handle_api_response(self, response: Dict[str, Any]) -> None:
        """处理API响应"""
        self.send_button.setEnabled(True)
        self.message_input.setReadOnly(False)

        if "choices" in response and len(response["choices"]) > 0:
            assistant_reply = response["choices"][0]["message"]["content"]
            self.conversation.add_assistant_message(assistant_reply)
            self.chat_history.append(f"<b>DeepSeek AI:</b> {assistant_reply}")

            # 滚动到底部
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.chat_history.setTextCursor(cursor)

            # 检查是否达到最大轮数
            if not self.conversation.can_continue():
                self.chat_history.append("<b>=== 已达到最大对话轮数 ===</b>")
        else:
            self.chat_history.append("<b>DeepSeek AI:</b> [无回复]")
            QMessageBox.warning(self, "API响应", "收到的响应格式不正确")

    def handle_api_error(self, error: str) -> None:
        """处理API错误"""
        self.send_button.setEnabled(True)
        self.message_input.setReadOnly(False)
        self.chat_history.append(f"<b>错误:</b> {error}")
        QMessageBox.critical(self, "API错误", f"请求失败: {error}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIChatApp()
    window.show()
    sys.exit(app.exec_())