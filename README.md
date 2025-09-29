# FashionAI 👗

  An AI-powered fashion assistant built with Streamlit that helps you discover and try on the latest fashion trends.

  ## Quick Start

  - Make the helper script executable (first time only):
    ```bash
    chmod +x run.sh
    ```
  - Run the app from the project root:
    ```bash
    ./run.sh
    ```
  This will create/activate a virtual environment, install dependencies, and start Streamlit at http://localhost:8501.

  Alternatively, you can run the app manually by following the instructions in the Setup section below.

  ## Features

  - Virtual Try-On
  - Outfit Recommendations
  - Style Analysis
  - Trend Forecasting

  ## Setup

  1. Clone the repository
  2. Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
  3. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

  ## Running the App

  ```bash
  streamlit run app.py
  ```

  ## Project Structure

  - `app.py` - Main application file
  - `requirements.txt` - Python dependencies
  - `README.md` - Project documentation
  - `.streamlit/config.toml` - Streamlit server and theme configuration
  - `run.sh` - One-command local run script

  ## Contributing

  Feel free to submit issues and enhancement requests.

  ## License

  MIT
