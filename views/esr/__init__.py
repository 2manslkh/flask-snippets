# ESR - Entity Sentiment Relationship
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify
)
import json
from werkzeug.exceptions import abort
from views.esr.text_preprocessing import create_json_output_single_file
from schemas.esr_schemas import validate_response

# ESR Blueprint
esr_bp = Blueprint(
    'esr', 
    __name__,
    template_folder='templates',
    static_folder='static'
    )

# ESR URLS
@esr_bp.route('/entity-sentiment-map')
def entity_sentiment():
    return render_template("esr/entity-sentiment-map.html", data="")

@esr_bp.route('/entity-sentiment-analysis', methods=['POST'])
def load_text():
    inp = request.form['free_text_input']
    # print(inp)
    output = create_json_output_single_file(inp)
    # print(type(output))
    validation = validate_response(output)

    if (validation['ok']) == True:
        flash("Loaded text successfully", validation['message'])
        return render_template("esr/entity-sentiment-map.html", data=json.dumps(output))
    else:
        flash("Processing is unsuccessful", validation['message'])
        return render_template("esr/entity-sentiment-map.html", data="")

@esr_bp.route('/entity-summary', methods=['GET'])
def query_db():
    result = ""
    return render_template("esr/entity-sentiment-summary.html", data=result)