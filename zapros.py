from flask import Flask, render_template, request, jsonify
from agent_invoking import agent_responses

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    question = request.json["message"]

    try:
        reply = agent_responses(question)
    except Exception as e:
        reply = str(e)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(port=5000, debug=True)