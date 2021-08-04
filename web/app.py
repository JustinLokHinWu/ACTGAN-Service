from flask import Flask, redirect, url_for
from flask import request, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate/cifar')
def generate_cifar():
    return 'cifar'

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')
