from flask import Flask, request, jsonify
import asyncio
import hypercorn.asyncio
from hypercorn.config import Config
from async_call_LLM import translate_text

app = Flask(__name__)


@app.route("/translate", methods=["POST"])
async def translate():
    data = request.get_json()
    translation = await translate_text(data)
    return jsonify(translation)

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:3001"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
