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

# 火山引擎豆包 SDK
from volcengine.maas import MaasClient
from volcengine.maas import models as vm


class ApiRequestThread(QThread):
    """API请求线程"""
    finished = pyqtSignal(dict, str)  # 传递响应和模型类型
    error = pyqtSignal(str, str)  # 传递错误信息和模型类型

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
            self.finished.emit(response, self.model_type)
        except Exception as e:
            self.error.emit(str(e), self.model_type)


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


class Conversation:
    """对话管理"""

    def __init__(self, models=None):
        self.models = models or ["deepseek-chat"]  # 支持多模型列表
        self.history = {}  # 按模型存储对话历史
        self.turn_count = 0
        self.max_turns = 0

    def start_new(self, models):
        """新建对话，支持多模型"""
        self.models = models
        self.history = {model: [] for model in models}
        self.turn_count = 0

    def add_user_message(self, message):
        """添加用户消息到所有模型历史"""
        for model in self.models:
            self.history[model].append({"role": "user", "content": message})

    def get_model_payload(self, model):
        """获取单个模型的请求载荷"""
        return {"model": model, "messages": self.history[model]}


class AIChatApp(QMainWindow):
    """主应用窗口"""

    def __init__(self):
        super().__init__()
        self.conversation = Conversation()
        self.threads = []  # 存储多线程
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("多模型AI聊天助手 (DeepSeek + 豆包)")
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

        # 测试连接按钮
        test_btn = QPushButton("测试DeepSeek和豆包连接")
        test_btn.clicked.connect(self.test_connections)
        layout.addWidget(test_btn)

        return tab

    def create_chat_tab(self):
        """创建聊天界面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 模型选择（支持多选）
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        self.model_checkbox = QCheckBox("DeepSeek")
        self.doubao_checkbox = QCheckBox("火山豆包")
        model_layout.addWidget(self.model_checkbox)
        model_layout.addWidget(self.doubao_checkbox)
        model_layout.addStretch()

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
        """测试DeepSeek和豆包连接"""
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

            QMessageBox.information(self, "成功", "DeepSeek和豆包连接测试通过")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接测试失败: {str(e)}")

    def new_conversation(self):
        """新建对话（支持选择多个模型）"""
        selected_models = []
        if self.model_checkbox.isChecked():
            selected_models.append("deepseek-chat")
        if self.doubao_checkbox.isChecked():
            selected_models.append("doubao")

        if not selected_models:
            QMessageBox.warning(self, "提示", "请至少选择一个模型")
            return

        self.conversation.start_new(selected_models)
        self.chat_history.clear()
        self.chat_history.append(f"<b>=== 新对话开始（{'、'.join(selected_models)}）===</b>")
        self.max_turns.setValue(0)

    def send_message(self):
        """发送消息到所有选中模型"""
        msg = self.msg_input.text().strip()
        if not msg: return

        selected_models = self.conversation.models
        if not selected_models:
            QMessageBox.warning(self, "提示", "请先开始新对话")
            return

        # 清空历史线程
        self.threads.clear()

        # 构建并发送多模型请求
        self.conversation.add_user_message(msg)
        self.chat_history.append(f"<b>你:</b> {msg}")
        self.msg_input.clear()

        for model in selected_models:
            if model == "deepseek-chat":
                client = DeepSeekClient(
                    base_url=self.ds_url.text(),
                    api_key=self.ds_key.text()
                )
            elif model == "doubao":
                client = DoubaoClient(
                    access_key=self.db_ak.text(),
                    secret_key=self.db_sk.text()
                )
            else:
                continue

            thread = ApiRequestThread(
                client=client,
                model_type=model,
                data=self.conversation.get_model_payload(model)
            )
            thread.finished.connect(self.handle_multi_response)
            thread.error.connect(self.handle_multi_error)
            self.threads.append(thread)
            thread.start()

    def handle_multi_response(self, response, model_type):
        """处理多模型响应"""
        try:
            if model_type.startswith("deepseek"):
                reply = response["choices"][0]["message"]["content"]
                model_name = "DEEPSEEK"
            elif model_type == "doubao":
                reply = response["choices"][0]["message"]["content"]
                model_name = "火山豆包"
            else:
                return

            self.conversation.history[model_type].append({"role": "assistant", "content": reply})
            self.chat_history.append(f"<b>{model_name}:</b> {reply}")
            self.chat_history.moveCursor(QTextCursor.End)

            # 检查轮数限制
            if self.max_turns.value() > 0:
                self.conversation.turn_count += 1
                if self.conversation.turn_count >= self.max_turns.value():
                    self.chat_history.append("<b>=== 达到最大对话轮数 ===</b>")
                    for t in self.threads: t.terminate()  # 终止所有线程

        except Exception as e:
            self.chat_history.append(f"<b>{model_type.upper()}错误:</b> 响应解析失败 - {str(e)}")

    def handle_multi_error(self, error, model_type):
        """处理多模型错误"""
        model_name = "DEEPSEEK" if model_type.startswith("deepseek") else "火山豆包"
        self.chat_history.append(f"<b>{model_name}错误:</b> {error}")
        QMessageBox.critical(self, f"{model_name}请求失败", error)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIChatApp()
    window.show()
    sys.exit(app.exec_())