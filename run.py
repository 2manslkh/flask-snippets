import os
from flask import Flask, render_template

app = Flask(__name__)
# app.secret_key = b'_5#asfasd"F4Q8z\n\xec]/'

@app.route('/')
@app.route('/<module_url>') # to include paths to itself
def index(module_url):
    module_data = {0:{"name":"Machine Learning",
                       "module_url":"machine_learning"},
                    1:{"name":"Data Visualization",
                        "module_url":"data_visualization"},
                    2:{"name":"Algorithms",
                        "module_url":"algorithms"},
                    3:{"name":"Deployment",
                        "module_url":"deployment"}
                }
    return render_template("home.html", modules=module_data)

# @app.route('/machine_learning')
# def hello2():
#     module_data = {0:{"name":"Machine Learning",
#                        "module_url":"machine_learning"},
#                     1:{"name":"Data Visualization",
#                         "module_url":"data_visualization"},
#                     2:{"name":"Algorithms",
#                         "module_url":"algorithms"},
#                     3:{"name":"Deployment",
#                         "module_url":"deployment"}
#                 }
#     return render_template("home.html", modules=module_data)

# set FLASK_APP=__init__.py
# set FLASK_ENV=development
if __name__ == "__main__":
    app.run(debug=True,port=5002)