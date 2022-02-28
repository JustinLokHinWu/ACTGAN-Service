from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from attrdict import AttrDict
import json
import io
import os

from ACTGAN import run_model
app = Flask(__name__)
CORS(app)

config = {}

@app.before_first_request
def load_configs():
    for filename in os.listdir('configs'):
        if filename.endswith('json'):
            with open('configs/' + filename) as f:
                key = filename.split('.')[0]
                config[key] = json.load(f)
    
    # app.logger.info(config)

def setup_model(dataset):
    generator_runner = None
    try:
        cfg = AttrDict(config[dataset]['model_params'])
        generator_runner = run_model.GeneratorRunner(cfg)
    except Exception:
        app.log_exception(Exception)
    return generator_runner

    # try:
    #     app.logger.info('Initiating generator runner')
    #     with open('./ACTGAN/run_config.json') as f:
    #         cfg = AttrDict(json.load(f))
    #         generator_runner = run_model.GeneratorRunner(cfg)
    #         app.logger.info('Initiated generator runner')
    # except Exception:
    #     app.log_exception(Exception)
    # return generator_runner

@app.route('/generate', methods=['POST'])
def generate():
    if request.is_json:
        req = request.get_json()

        # Get class id, error on missing 
        if 'class_id' in req:
            class_id = req['class_id']
        else:
            return 'Missing class id', 400

        # Get dataset
        if 'dataset' in req:
            dataset = req['dataset']
        else:
            return 'Invalid dataset', 400

        # Setup correct generator
        if dataset in config:
            generator_runner = setup_model(dataset)
        else:
            return 'Invalid dataset', 400

        # Get epochs from request, else use most recent epoch
        valid_epochs = generator_runner.get_valid_epochs()
        if 'epoch' in req:
            epoch = req['epoch']
            if epoch not in valid_epochs:
                return 'Invalid epoch', 400
        else:
            epoch = valid_epochs[-1]

        # Get seed, optional
        if 'seed' in req:
            seed = req['seed']
        else:
            seed = None
    else:
        return 'Request is not json', 400

    # Generate image using request parameters
    result = generator_runner.evaluate(class_id, epoch, seed)
    if result is None:
        return 'Image generation failed', 500
    
    # Convert PILImage to JPEG, return to client 
    result_io = io.BytesIO()
    result.save(result_io, 'JPEG', quality=70)
    result_io.seek(0)

    return send_file(result_io, mimetype='image/jpeg')

@app.route('/get-epochs', methods=['GET'])
def get_epochs():
    dataset = request.args.get('dataset')
    if dataset is not None:
        # Setup correct generator
        if dataset in config:
            generator_runner = setup_model(dataset)
        else:
            return 'Invalid dataset', 400

        # Get valid epochs for this dataset
        valid_epochs = generator_runner.get_valid_epochs()
        valid_epochs.reverse()
        return jsonify(valid_epochs)

    else:
        return 'Missing dataset parameter', 400

@app.route('/get-classes', methods=['GET'])
def get_classes():
    dataset = request.args.get('dataset')
    if dataset is not None:
        # Setup correct generator
        if dataset in config:
            return jsonify(config[dataset]['classes'])
        else:
            return 'Invalid dataset', 400
    else:
        return 'Missing dataset parameter', 400

@app.route('/get-datasets', methods=['GET'])
def get_datasets():
    return jsonify(list(config.keys()))

def create_app():
    print("Starting production server")
    return app

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')
