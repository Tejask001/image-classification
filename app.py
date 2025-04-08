from flask import Flask, request, jsonify
import requests
import os
import subprocess
import tempfile
import json
import re  # Import the 're' module

app = Flask(__name__)


@app.route('/process_image', methods=['POST'], )
def process_image():
    try:
        data = request.get_json()  # Get JSON data from the request

        if not data or 'image_url' not in data:
            return jsonify({'error': 'Missing or invalid request body.  '
                                     'Please provide a JSON object with an "image_url" key.'}), 400

        image_url = data['image_url']

        # Download the image
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Error downloading image: {str(e)}'}), 400

        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_url)[1]) as temp_image:  #added suffix from url
            for chunk in response.iter_content(chunk_size=8192):
                temp_image.write(chunk)
            image_path = temp_image.name  # Store the file path

        # Run the script.py with the image path
        try:
            result = subprocess.run(['python3', 'script.py', image_path],
                                    capture_output=True, text=True, check=True)
            output = result.stdout
            error = result.stderr

            # Log any errors for debugging
            if error:
                print(f"Script.py error: {error}")

        except subprocess.CalledProcessError as e:
            # Capture subprocess errors (e.g., script.py failing)
            return jsonify({'error': f'Error running script.py: {str(e)}, output: {e.stdout}, error: {e.stderr}'}), 500
        except FileNotFoundError:
            return jsonify({'error': 'script.py not found.  Make sure it is in the same directory as this API script.'}), 500
        except Exception as e:
            return jsonify({'error': f'Unexpected error running script.py: {str(e)}'}), 500

        try:
            # Find JSON within output (robust approach)
            match = re.search(r"\{.*\}", output, re.DOTALL)  # Find JSON-like string
            if match:
                json_string = match.group(0)  # Extract matched JSON string
                # Use json.loads() to parse the JSON string
                output_dict = json.loads(json_string)
            else:
                return jsonify({'error': 'No JSON found in script.py output'}), 500
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Error parsing JSON: {str(e)}. Output was: {output}'}), 500

        finally:
            os.remove(image_path)  # Ensure temp file is cleaned up

        return jsonify(output_dict), 200

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(port=3000)