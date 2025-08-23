# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gradio as gr
from llamafactory.webui.interface import create_ui

def main():
    # 获取环境变量或设置默认值
    gradio_ipv6 = os.getenv("GRADIO_IPV6", "0").lower() in ["true", "1"]
    gradio_share = os.getenv("GRADIO_SHARE", "0").lower() in ["true", "1"]
    server_name = "127.0.0.1"
    server_port = 7888  # 默认端口
    #server_port = 6900

    # 调试 create_ui()
    try:
        ui = create_ui()
        if not isinstance(ui, (gr.Blocks, gr.Interface)):
            raise ValueError("create_ui() did not return a valid Gradio object.")
        print("UI successfully created.")
    except Exception as e:
        print(f"Error creating UI: {e}")
        return

    # 启动 Gradio 应用
    try:
        output = ui.queue().launch(share=True, server_name=server_name, server_port=server_port, inbrowser=True)
        print(f"Public link: {output}")
    except Exception as e:
        print(f"Error launching Gradio app: {e}")

if __name__ == "__main__":
    main()


