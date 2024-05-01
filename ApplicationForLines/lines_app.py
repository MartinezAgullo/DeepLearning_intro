from flask import Flask, render_template, request, send_file
import matplotlib  # Ensure this is defined
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io  # to create an in-memory buffer for the plot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Renders the form to input coefficients

@app.route('/plot', methods=['POST'])
def plot():
    app.logger.debug("Received plot request.")
    # Retrieve coefficients from the form submission
    a = float(request.form['a'])
    b = float(request.form['b'])
    c = float(request.form['c'])
    
    # Create the line plot
    x = np.linspace(-10, 10, 100)
    y = (-a * x - c) / b
    
    plt.figure()
    plt.plot(x, y, label=f'{a}x + {b}y + {c} = 0')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Line Plot from Implicit Equation')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # x-axis
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # y-axis
    plt.legend()
    
    # Store the plot in memory and send it as an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Reset the buffer position
    plt.close()  # Close the plot to avoid memory leaks
    
    # Return the plot as a PNG image
    return send_file(buf, mimetype='image/png')
