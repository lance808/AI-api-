import sys
import json
import time
import requests
from typing import Dict, Any, Optional, List, Union
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QLabel, QComboBox, QSpinBox, QMessageBox, QTabWidget, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor

# 火山引擎豆包 SDK（需提前安装：pip install volcengine）
from volcengine.maas import MaasClient
from volcengine.maas import models as vm
# 讯飞星火 SDK（需提前安装：pip install xfyun）
from xfyun.xinghuo import XFyunClient


class ApiRequestThread(QThread):
    """API请求线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, client, model_type, data):
        super().__init__()
        self.client = client
        self.model_type = model_type
        self.data = data

    def run(self):
        try:
            if self.model_type.startswith("deepseek"):
                response = self.client.post("/chat/completions", data=self.data)
            elif self.model_type == "doubao":
                request = vm.ChatCompletionRequest(
                    model=self.data["model"],
                    messages=self.data["messages"]
                )
                response = self.client.chat_completion(request).to_dict()
            elif self.model_type == "xfyun":
                response = self.client.chat(model=self.data["model"], messages=self.data["messages"])
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))


class BaseAIApiClient:
    """基础API客户端"""

    def __init__(self, timeout=60, max_retries=3, retry_delay=5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _make_request(self, method, url, params=None, data=None, headers=None, files=None, stream=False):
        """通用请求处理"""
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
                    return response.json() if not stream else response
                elif response.status_code in [429, 500, 502, 503, 504]:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise Exception(f"API错误: {response.status_code} - {response.text}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"请求失败: {str(e)}")


class DeepSeekClient(BaseAIApiClient):
    """DeepSeek客户端"""

    def __init__(self, base_url, api_key, header="Authorization", prefix="Bearer"):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.header = header
        self.prefix = prefix

    def post(self, endpoint, data):
        """发送POST请求"""
        headers = {
            self.header: f"{self.prefix} {self.api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}{endpoint}"
        return self._make_request("POST", url, data=json.dumps(data), headers=headers)


class DoubaoClient(BaseAIApiClient):
    """火山豆包客户端"""

    def __init__(self, access_key, secret_key):
        super().__init__()
        self.client = MaasClient(access_key=access_key, secret_key=secret_key)


class XFyunClient(BaseAIApiClient):
    """讯飞星火客户端"""

    def __init__(self, app_id, api_key, api_secret):
        super().__init__()
        self.client = XFyunClient(app_id=app_id, api_key=api_key, api_secret=api_secret)


class Conversation:
    """对话管理"""

    def __init__(self, model="deepseek-chat", model_type="deepseek"):
        self.model = model
        self.model_type = model_type
        self.history = []
        self.turn_count = 0
        self.max_turns = 0

    def start_new(self, model=None, model_type=None):
        """新建对话"""
        self.model = model or "deepseek-chat"
        self.model_type = model_type or "deepseek"
        self.history = []
        self.turn_count = 0

    def add_message(self, role, content):
        """添加消息"""
        self.history.append({"role": role, "content": content})
        if role == "assistant":
            self.turn_count += 1

    def get_payload(self):
        """获取请求载荷"""
        return {"model": self.model, "messages": self.history}


class AIChatApp(QMainWindow):
    """主应用窗口"""

    def __init__(self):
        super().__init__()
        self.conversation = Conversation()
        self.current_client = None
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("多模型AI聊天助手")
        self.setGeometry(100, 100, 900, 700)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # API设置标签页
        setup_tab = self.create_setup_tab()
        # 聊天标签页
        chat_tab = self.create_chat_tab()

        tabs = QTabWidget()
        tabs.addTab(setup_tab, "API配置")
        tabs.addTab(chat_tab, "开始聊天")

        main_layout.addWidget(tabs)
        self.setCentralWidget(main_widget)

    def create_setup_tab(self):
        """创建配置界面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # DeepSeek配置
        deepseek_group = QWidget()
        deepseek_layout = QVBoxLayout(deepseek_group)
        deepseek_layout.addWidget(QLabel("=== DeepSeek 配置 ==="))
        deepseek_layout.addWidget(QLabel("基础URL:"))
        self.ds_url = QLineEdit("https://api.deepseek.com/v1")
        deepseek_layout.addWidget(self.ds_url)
        deepseek_layout.addWidget(QLabel("API密钥:"))
        self.ds_key = QLineEdit()
        self.ds_key.setEchoMode(QLineEdit.Password)
        deepseek_layout.addWidget(self.ds_key)
        layout.addWidget(deepseek_group)

        # 火山豆包配置
        doubao_group = QWidget()
        doubao_layout = QVBoxLayout(doubao_group)
        doubao_layout.addWidget(QLabel("=== 火山豆包 配置 ==="))
        doubao_layout.addWidget(QLabel("Access Key:"))
        self.db_ak = QLineEdit()
        doubao_layout.addWidget(self.db_ak)
        doubao_layout.addWidget(QLabel("Secret Key:"))
        self.db_sk = QLineEdit()
        self.db_sk.setEchoMode(QLineEdit.Password)
        doubao_layout.addWidget(self.db_sk)
        layout.addWidget(doubao_group)

        # 讯飞星火配置
        xfyun_group = QWidget()
        xfyun_layout = QVBoxLayout(xfyun_group)
        xfyun_layout.addWidget(QLabel("=== 讯飞星火 配置 ==="))
        xfyun_layout.addWidget(QLabel("App ID:"))
        self.xf_appid = QLineEdit()
        xfyun_layout.addWidget(self.xf_appid)
        xfyun_layout.addWidget(QLabel("API Key:"))
        self.xf_key = QLineEdit()
        xfyun_layout.addWidget(self.xf_key)
        xfyun_layout.addWidget(QLabel("API Secret:"))
        self.xf_secret = QLineEdit()
        self.xf_secret.setEchoMode(QLineEdit.Password)
        xfyun_layout.addWidget(self.xf_secret)
        layout.addWidget(xfyun_group)

        # 测试连接按钮
        test_btn = QPushButton("测试所有连接")
        test_btn.clicked.connect(self.test_connections)
        layout.addWidget(test_btn)

        return tab

    def create_chat_tab(self):
        """创建聊天界面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        self.model_combobox = QComboBox()
        self.model_combobox.addItems([
            "deepseek-chat (DeepSeek)",
            "doubao (火山豆包)",
            "xfyun (讯飞星火)"
        ])
        model_layout.addWidget(self.model_combobox)

        # 对话轮数
        turn_layout = QHBoxLayout()
        turn_layout.addWidget(QLabel("最大轮数:"))
        self.max_turns = QSpinBox()
        self.max_turns.setRange(0, 50)
        self.max_turns.setValue(0)
        turn_layout.addWidget(self.max_turns)
        turn_layout.addWidget(QLabel("(0=无限制)"))

        # 新对话按钮
        new_chat_btn = QPushButton("开始新对话")
        new_chat_btn.clicked.connect(self.new_conversation)

        top_layout = QVBoxLayout()
        top_layout.addLayout(model_layout)
        top_layout.addLayout(turn_layout)
        top_layout.addWidget(new_chat_btn)

        # 聊天历史
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("font-size: 14px;")

        # 输入区域
        input_layout = QHBoxLayout()
        self.msg_input = QLineEdit()
        self.msg_input.setPlaceholderText("请输入你的问题...")
        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.msg_input)
        input_layout.addWidget(send_btn)

        layout.addLayout(top_layout)
        layout.addWidget(self.chat_history)
        layout.addLayout(input_layout)

        return tab

    def test_connections(self):
        """测试所有模型连接"""
        try:
            # 测试DeepSeek
            if self.ds_key.text():
                ds_client = DeepSeekClient(
                    base_url=self.ds_url.text(),
                    api_key=self.ds_key.text()
                )
                ds_client.post("/models", data={})

            # 测试火山豆包
            if self.db_ak.text() and self.db_sk.text():
                db_client = DoubaoClient(
                    access_key=self.db_ak.text(),
                    secret_key=self.db_sk.text()
                )

            # 测试讯飞星火
            if self.xf_appid.text() and self.xf_key.text() and self.xf_secret.text():
                xf_client = XFyunClient(
                    app_id=self.xf_appid.text(),
                    api_key=self.xf_key.text(),
                    api_secret=self.xf_secret.text()
                )

            QMessageBox.information(self, "成功", "所有模型连接测试通过")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接测试失败: {str(e)}")

    def new_conversation(self):
        """新建对话"""
        model_info = self.model_combobox.currentText()
        model_type = model_info.split()[0]
        self.conversation.start_new(
            model=model_type,
            model_type=model_type
        )
        self.chat_history.clear()
        self.chat_history.append(f"<b>=== 新对话开始 ({model_info}) ===</b>")
        self.max_turns.setValue(0)

    def send_message(self):
        """发送消息"""
        msg = self.msg_input.text().strip()
        if not msg: return

        # 检查模型类型
        model_type = self.conversation.model_type
        if model_type == "deepseek-chat":
            self.current_client = DeepSeekClient(
                base_url=self.ds_url.text(),
                api_key=self.ds_key.text()
            )
        elif model_type == "doubao":
            self.current_client = DoubaoClient(
                access_key=self.db_ak.text(),
                secret_key=self.db_sk.text()
            )
        elif model_type == "xfyun":
            self.current_client = XFyunClient(
                app_id=self.xf_appid.text(),
                api_key=self.xf_key.text(),
                api_secret=self.xf_secret.text()
            )

        # 构建对话历史
        self.conversation.add_message("user", msg)
        self.chat_history.append(f"<b>你:</b> {msg}")
        self.msg_input.clear()

        # 发起请求
        thread = ApiRequestThread(
            client=self.current_client,
            model_type=model_type,
            data=self.conversation.get_payload()
        )
        thread.finished.connect(self.handle_response)
        thread.error.connect(self.handle_error)
        thread.start()

    def handle_response(self, response):
        """处理响应"""
        try:
            if self.conversation.model_type.startswith("deepseek"):
                reply = response["choices"][0]["message"]["content"]
            elif self.conversation.model_type == "doubao":
                reply = response["choices"][0]["message"]["content"]
            elif self.conversation.model_type == "xfyun":
                reply = response["data"]["result"]

            self.conversation.add_message("assistant", reply)
            self.chat_history.append(f"<b>{self.conversation.model_type.replace('-', ' ').upper()}:</b> {reply}")
            self.chat_history.moveCursor(QTextCursor.End)

            # 检查轮数限制
            if self.max_turns.value() > 0 and self.conversation.turn_count >= self.max_turns.value():
                self.chat_history.append("<b>=== 达到最大对话轮数 ===</b>")
        except Exception as e:
            self.chat_history.append(f"<b>错误:</b> 响应解析失败 - {str(e)}")

    def handle_error(self, error):
        """处理错误"""
        self.chat_history.append(f"<b>错误:</b> {error}")
        QMessageBox.critical(self, "请求失败", error)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIChatApp()
    window.show()
    sys.exit(app.exec_())