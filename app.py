
from pathlib import Path

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Update this path to the location of your existing model file.
MODEL_PATH = Path(
    r"C:\Users\USER\Documents\ML_Python\california.joblib"
)

# Load the saved model and feature names
obj = joblib.load(MODEL_PATH)

model = obj["Model"]
columns = list(obj["Columns"])


@app.route("/", methods=["GET", "POST"])
def main():
    prediction = None
    error = None
    values = {}

    if request.method == "POST":
        input_values = []

        try:
            for column in columns:
                raw_value = request.form.get(column, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter a value for {column}."
                    )

                value = float(raw_value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"{column} must be a valid finite number."
                    )

                values[column] = raw_value
                input_values.append(value)

            result = model.predict([input_values])[0]
            prediction = float(result)

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid numeric values."
        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Unable to generate a prediction. "
                "Please check your inputs and try again."
            )

    return render_template(
        "index.html",
        columns=columns,
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)