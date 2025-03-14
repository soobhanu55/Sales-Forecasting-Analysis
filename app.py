from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    try:
        # Get form data with validation
        form_data = {
            'item_weight': float(request.form.get('item_weight', 0)),
            'item_fat_content': float(request.form.get('item_fat_content', 0)),
            'item_visibility': float(request.form.get('item_visibility', 0)),
            'item_type': float(request.form.get('item_type', 0)),
            'item_mrp': float(request.form.get('item_mrp', 0)),
            'outlet_establishment_year': float(request.form.get('outlet_establishment_year', 0)),
            'outlet_size': float(request.form.get('outlet_size', 0)),
            'outlet_location_type': float(request.form.get('outlet_location_type', 0)),
            'outlet_type': float(request.form.get('outlet_type', 0))
        }

        # Create input array
        X = np.array([[
            form_data['item_weight'],
            form_data['item_fat_content'],
            form_data['item_visibility'],
            form_data['item_type'],
            form_data['item_mrp'],
            form_data['outlet_establishment_year'],
            form_data['outlet_size'],
            form_data['outlet_location_type'],
            form_data['outlet_type']
        ]])

        # Get the current directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths to model files
        scaler_path = os.path.join(current_dir, 'models', 'sc.sav')
        model_path = os.path.join(current_dir, 'models', 'lr.sav')

        # Load and apply the scaler
        sc = joblib.load(scaler_path)
        X_std = sc.transform(X)
        
        # Load and apply the model
        model = joblib.load(model_path)
        Y_pred = model.predict(X_std)
        
        # Format the prediction with 2 decimal places
        prediction = round(float(Y_pred[0]), 2)
        
        # Create a dictionary with all the input values and prediction
        result_data = {
            'item_weight': form_data['item_weight'],
            'item_fat_content': form_data['item_fat_content'],
            'item_visibility': form_data['item_visibility'],
            'item_type': form_data['item_type'],
            'item_mrp': form_data['item_mrp'],
            'outlet_establishment_year': form_data['outlet_establishment_year'],
            'outlet_size': form_data['outlet_size'],
            'outlet_location_type': form_data['outlet_location_type'],
            'outlet_type': form_data['outlet_type'],
            'prediction': prediction
        }
        
        return render_template('result.html', result=result_data)
        
    except ValueError as e:
        return render_template('error.html', error='Please enter valid numeric values for all fields.')
    except FileNotFoundError as e:
        return render_template('error.html', error=f'Model file not found: {str(e)}')
    except Exception as e:
        return render_template('error.html', error=f'An error occurred: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=9457)
