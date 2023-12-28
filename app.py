import os
from Utilities import Utils
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = load_model("Cataract-Model")

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

util = Utils()
classes = util.class_names 

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/in")
def helloin():
    return render_template("compare.html")

@app.route("/test",  methods = ['GET' , 'POST'])
def test():
    model_name = request.form.get('models')
    if model_name == 'cm':
        return render_template("compare.html")

    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                # class_result , prob_result = util.make_predictions(model_name, img_path)
                prob_result , class_result = util.make_predictions(model_name, img_path)

                # predictions = (class_result , int(prob_result*100))
                predictions = (class_result , prob_result)

            file_name_m = f'static/graphs/{model_name}/mat.png'
            file_name_f = f'static/graphs/{model_name}/f.png'
            file_name_metrics = f'static/graphs/{model_name}/metrics.png'

            return render_template("success.html", img=img, predictions=predictions, name=model_name, file_name_m=file_name_m, file_name_f=file_name_f, file_name_metrics=file_name_metrics)
        else :
            return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)