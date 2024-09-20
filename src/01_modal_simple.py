# 一時的な関数をクラウド上で実行するサンプル
# ref: https://modal.com/docs/examples/hello_world

import modal

app = modal.App("example-hello-world")

# Function の定義. この関数はクラウド上で実行することができる
@app.function()
def square(i):
    return i * i

# ローカルから実行するためのエントリポイント
@app.local_entrypoint()
def main():
    print(square.remote(1000))

"""example:
uv run modal deploy src.01_modal_simple
uv run modal run src.01_modal_simple
"""
