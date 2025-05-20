import sys
import json
import time
import os
import requests
from typing import Dict, Any, Optional, List, Union
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QLabel, QComboBox, QSpinBox, QMessageBox, QTabWidget,
                             QCheckBox, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor, QColor
from volcenginesdkarkruntime import Ark  # 确保已安装此SDK


# ====================== 增强的API客户端 ======================
class BaseAIClient(QObject):
    """基础AI客户端抽象类"""
    stream_update = pyqtSignal(str, str)  # (content, model_name)
    error_occurred = pyqtSignal(str, str)  # (error_msg, model_name)


class DeepSeekClient(BaseAIClient):
    """DeepSeek客户端支持流式和非流式"""

    def __init__(self, config):
        super().__init__()
        self.base_url = config['base_url']
        self.api_key = config['api_key']
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, data, stream=False):
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.lstrip(b'data: ').decode('utf-8'))
                            if 'choices' in chunk:
                                content = chunk['choices'][0]['delta'].get('content', '')
                                if content:
                                    self.stream_update.emit(content, "DeepSeek")
                        except Exception as e:
                            continue
            else:
                result = response.json()
                return result['choices'][0]['message']['content']
        except Exception as e:
            self.error_occurred.emit(str(e), "DeepSeek")


class DoubaoClient(BaseAIClient):
    """火山豆包客户端支持流式和非流式"""

    def __init__(self, config):
        super().__init__()
        self.client = Ark(api_key=config['api_key'])
        self.model = config.get('model', 'doubao-1-5-pro-32k-250115')

    def generate(self, data, stream=False):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=data['messages'],
                stream=stream,
                stream_options={"include_usage": True}
            )

            full_response = ""
            for chunk in resp:
                content = ""
                if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                    content += chunk.choices[0].delta.reasoning_content or ""
                if hasattr(chunk.choices[0].delta, 'content'):
                    content += chunk.choices[0].delta.content or ""

                if content:
                    if stream:
                        self.stream_update.emit(content, "火山豆包")
                    else:
                        full_response += content
            return full_response
        except Exception as e:
            self.error_occurred.emit(str(e), "火山豆包")


# ====================== 更新后的线程类 ======================
class StreamRequestThread(QThread):
    """流式请求线程"""
    update_signal = pyqtSignal(str, str)  # (content, model_name)
    finished = pyqtSignal(str, str)  # (full_response, model_name)
    error_signal = pyqtSignal(str, str)  # (error_msg, model_name)

    def __init__(self, client, data, model_name, stream=False):
        super().__init__()
        self.client = client
        self.data = data
        self.model_name = model_name
        self.stream = stream

    def run(self):
        try:
            if self.stream:
                self.client.stream_update.connect(self.update_signal)
                self.client.error_occurred.connect(self.error_signal)
                self.client.generate(self.data, stream=True)
            else:
                response = self.client.generate(self.data)
                self.finished.emit(response, self.model_name)
        except Exception as e:
            self.error_signal.emit(str(e), self.model_name)


# ====================== 更新主界面 ======================
class AIChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.clients = {}
        self.config = {
            'deepseek': {'base_url': 'https://api.deepseek.com/v1', 'api_key': ''},
            'doubao': {'api_key': ''}
        }
        self.active_threads = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("多模型AI聊天助手")
        self.setGeometry(100, 100, 1000, 800)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 配置选项卡
        self.tabs = QTabWidget()
        self.setup_config_tabs()

        # 聊天界面
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)

        # 控制栏
        control_bar = QHBoxLayout()
        self.model_checks = {
            'deepseek': QCheckBox("DeepSeek"),
            'doubao': QCheckBox("火山豆包")
        }
        self.stream_check = QCheckBox("流式响应")
        self.new_chat_btn = QPushButton("新对话")
        self.send_btn = QPushButton("发送")

        for cb in self.model_checks.values():
            control_bar.addWidget(cb)
        control_bar.addWidget(self.stream_check)
        control_bar.addWidget(self.new_chat_btn)
        control_bar.addWidget(self.send_btn)

        # 聊天历史
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("font-size: 14px;")

        # 输入框
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入消息...")

        # 布局
        chat_layout.addLayout(control_bar)
        chat_layout.addWidget(self.chat_history)
        chat_layout.addWidget(self.input_box)

        self.tabs.addTab(chat_widget, "聊天")
        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

        # 信号连接
        self.send_btn.clicked.connect(self.send_message)
        self.new_chat_btn.clicked.connect(self.new_chat)

    def setup_config_tabs(self):
        # DeepSeek配置
        ds_tab = QWidget()
        ds_layout = QVBoxLayout(ds_tab)
        ds_layout.addWidget(QLabel("API Base URL:"))
        self.ds_url = QLineEdit(self.config['deepseek']['base_url'])
        ds_layout.addWidget(self.ds_url)
        ds_layout.addWidget(QLabel("API Key:"))
        self.ds_key = QLineEdit()
        self.ds_key.setEchoMode(QLineEdit.Password)
        ds_layout.addWidget(self.ds_key)
        self.tabs.addTab(ds_tab, "DeepSeek设置")

        # 豆包配置
        db_tab = QWidget()
        db_layout = QVBoxLayout(db_tab)
        db_layout.addWidget(QLabel("API Key:"))
        self.db_key = QLineEdit()
        self.db_key.setEchoMode(QLineEdit.Password)
        db_layout.addWidget(self.db_key)
        self.tabs.addTab(db_tab, "火山豆包设置")

    def new_chat(self):
        self.chat_history.clear()
        self.clients.clear()
        self.active_threads = []

    def send_message(self):
        message = self.input_box.text().strip()
        if not message:
            return

        selected_models = [k for k, v in self.model_checks.items() if v.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "错误", "请至少选择一个模型")
            return

        # 保存当前配置
        self.config['deepseek']['api_key'] = self.ds_key.text()
        self.config['doubao']['api_key'] = self.db_key.text()

        # 初始化客户端
        clients = {}
        if 'deepseek' in selected_models:
            clients['deepseek'] = DeepSeekClient(self.config['deepseek'])
        if 'doubao' in selected_models:
            clients['doubao'] = DoubaoClient(self.config['doubao'])

        # 显示用户消息
        self.chat_history.append(f"<b>用户：</b>{message}")
        self.input_box.clear()

        # 为每个模型创建线程
        for model_name in selected_models:
            client = clients[model_name]
            data = {
                "messages": [{"role": "user", "content": message}],
                "model": self.config[model_name].get('model', '')
            }

            thread = StreamRequestThread(
                client=client,
                data=data,
                model_name=model_name,
                stream=self.stream_check.isChecked()
            )

            if self.stream_check.isChecked():
                thread.update_signal.connect(self.update_stream_response)
            else:
                thread.finished.connect(self.show_full_response)

            thread.error_signal.connect(self.handle_error)
            thread.start()
            self.active_threads.append(thread)

    def update_stream_response(self, content, model_name):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)

        # 查找现有模型响应位置
        model_tag = f"[{model_name}]"
        last_block = self.chat_history.document().lastBlock().text()

        if model_tag not in last_block:
            self.chat_history.append(f"<b>{model_tag}</b>: ")
            cursor.movePosition(QTextCursor.End)

        cursor.insertText(content)
        self.chat_history.setTextCursor(cursor)

    def show_full_response(self, response, model_name):
        self.chat_history.append(f"<b>[{model_name}]</b>: {response}")

    def handle_error(self, error, model_name):
        self.chat_history.append(f"<font color='red'><b>{model_name}错误：</b>{error}</font>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIChatApp()
    window.show()
    sys.exit(app.exec_())