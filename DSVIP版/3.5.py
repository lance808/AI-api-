import sys
import os
import json
import time
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QLabel, QCheckBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from volcenginesdkarkruntime import Ark  # 火山豆包官方SDK
import requests  # DeepSeek HTTP调用


class StreamRequestThread(QThread):
    """流式请求线程"""
    update_signal = pyqtSignal(str, str)  # (增量内容, 模型类型)
    error_signal = pyqtSignal(str, str)  # (错误信息, 模型类型)

    def __init__(self, model_type, config, message, is_stream):
        super().__init__()
        self.model_type = model_type
        self.config = config
        self.message = message
        self.is_stream = is_stream

    def run(self):
        try:
            if self.model_type == "doubao":
                self.handle_doubao_stream()
            elif self.model_type == "deepseek":
                self.handle_deepseek_stream()

        except Exception as e:
            self.error_signal.emit(str(e), self.model_type)

    def handle_doubao_stream(self):
        """处理火山豆包流式响应"""
        client = Ark(api_key=self.config["api_key"])
        resp = client.chat.completions.create(
            model=self.config["model"],
            messages=[{"role": "user", "content": self.message}],
            stream=self.is_stream,
            stream_options={"include_usage": True}
        )

        for chunk in resp:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = ""
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                content += delta.reasoning_content
            if hasattr(delta, 'content') and delta.content:
                content += delta.content

            if content:
                self.update_signal.emit(content, "火山豆包")

    def handle_deepseek_stream(self):
        """处理DeepSeek流式响应（需模型支持）"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": self.message}],
            "stream": self.is_stream  # DeepSeek是否支持流式需确认文档
        }

        response = requests.post(
            f"{self.config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.lstrip(b'data: '))
                    content = json_data["choices"][0]["delta"]["content"]
                    self.update_signal.emit(content, "DeepSeek")
                except Exception as e:
                    pass


class MultiModelChatUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.config = {
            "doubao": {
                "api_key": os.environ.get("ARK_API_KEY", ""),
                "model": "doubao-1-5-pro-32k-250115",
                "base_url": "https://api.volcengine.com/maas/v1/chat"
            },
            "deepseek": {
                "api_key": "",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat-mix-12b"
            }
        }
        self.is_stream = False  # 默认非流式

    def init_ui(self):
        self.setWindowTitle("多模型流式聊天工具")
        self.setGeometry(100, 100, 800, 700)

        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)

        # 模型选择与流式开关
        control_layout = QHBoxLayout()
        self.model_checkboxes = {
            "doubao": QCheckBox("火山豆包"),
            "deepseek": QCheckBox("DeepSeek")
        }
        self.stream_switch = QCheckBox("启用流式响应")
        control_layout.addWidget(QLabel("选择模型："))
        for cb in self.model_checkboxes.values():
            control_layout.addWidget(cb)
        control_layout.addWidget(self.stream_switch)
        control_layout.addStretch()

        # 输入框
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("请输入你的问题...")
        self.input_box.setStyleSheet("font-size: 14px; height: 36px;")

        # 发送按钮
        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self.send_message)
        send_btn.setStyleSheet("font-size: 14px; padding: 6px 20px;")

        # 回答显示区域
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            font-size: 14px; 
            line-height: 1.6; 
            color: #333; 
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        """)

        layout.addLayout(control_layout)
        layout.addWidget(self.input_box)
        layout.addWidget(send_btn)
        layout.addWidget(self.output_area)
        self.setCentralWidget(main_widget)

    def send_message(self):
        question = self.input_box.text().strip()
        if not question:
            return

        selected_models = [mt for mt, cb in self.model_checkboxes.items() if cb.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "提示", "请至少选择一个模型")
            return

        self.is_stream = self.stream_switch.isChecked()
        self.output_area.append(f"<b>[{time.strftime('%H:%M:%S')}]</b> 你: {question}")
        self.input_box.clear()

        for model in selected_models:
            thread = StreamRequestThread(
                model_type=model,
                config=self.config[model],
                message=question,
                is_stream=self.is_stream
            )
            thread.update_signal.connect(self.update_output)
            thread.error_signal.connect(self.handle_error)
            thread.start()

    def update_output(self, content, model_name):
        """实时更新输出区域"""
        self.output_area.moveCursor(QTextCursor.End)
        self.output_area.insertHtml(f"<b>{model_name}:</b> {content}")

    def handle_error(self, error, model_name):
        """错误处理"""
        self.output_area.append(f"\n<b>{model_name}错误:</b> {error}\n")
        QMessageBox.critical(self, f"{model_name}调用失败", error, buttons=QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiModelChatUI()
    window.show()
    sys.exit(app.exec_())