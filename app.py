from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    # Direct HTML response instead of using templates
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Home</title>
    </head>
    <body>
        <h1>Welcome to the Home Page!</h1>
        <p>This is a direct HTML response, not using Jinja templates.</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
