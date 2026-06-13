from flask import request, jsonify, send_file
from module4.rag_pipeline import prepare_index, answer_query
from module4.highlight import draw_bbox

import json


def register_module4_routes(app):

    @app.route("/index_doc", methods=["POST"])
    def index_doc():
        data = request.json

        grounding = data["grounding"]
        prepare_index(grounding)

        return jsonify({"status": "indexed"})


    @app.route("/query", methods=["POST"])
    def query():
        query = request.json["query"]
        image_path = request.json["image"]

        result = answer_query(query)

        highlighted = draw_bbox(image_path, result["bbox"])

        return jsonify({
            "answer": result["answer"],
            "highlight_image": highlighted
        })