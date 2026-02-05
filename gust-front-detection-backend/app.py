"""Dummy version of the app that is just being used as a proof of concept for the docker"""


from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Docker World!'

if __name__ == '__main__':
    # Listen on all available network interfaces (0.0.0.0)
    app.run(debug=True, host='0.0.0.0', port=5000)

