from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from attrdict import AttrDict
import json
import io

from ACTGAN import run_model
app = Flask(__name__)
CORS(app)

config_mapping = {
    'cifar': './configs/cifar.json'
}

def setup_model(cfg_path):
    generator_runner = None
    try:
        app.logger.info('Initiating generator runner')
        with open('./ACTGAN/run_config.json') as f:
            cfg = AttrDict(json.load(f))
            generator_runner = run_model.GeneratorRunner(cfg)
            app.logger.info('Initiated generator runner')
    except Exception:
        app.log_exception(Exception)
    return generator_runner

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
            return 'Missing dataset', 400

        # Setup correct generator
        if dataset == 'cifar':
            generator_runner = setup_model(config_mapping['cifar'])
        elif dataset == 'mnist':
            return 'Datset not yet implemented', 400
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
    if request.is_json:
        req = request.get_json()

        # Get dataset
        if 'dataset' in req:
            dataset = req['dataset']
        else:
            return 'Missing dataset', 400

        # Setup correct generator
        if dataset == 'cifar':
            generator_runner = setup_model(config_mapping['cifar'])
        elif dataset == 'mnist':
            return 'Datset not yet implemented', 400
        else:
            return 'Invalid dataset', 400

        # Get valid epochs for this dataset
        valid_epochs = generator_runner.get_valid_epochs()
        return jsonify({'epochs': valid_epochs})

    else:
        return 'Request is not json', 400


if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')
