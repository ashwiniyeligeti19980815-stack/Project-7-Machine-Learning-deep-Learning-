# Disaster Tweet Classifier

Flask application serving the project's TF-IDF and Multinomial Naive Bayes classifier.

## Train and run locally

Place the labeled dataset in the project directory. It must be an Excel or CSV file with `text` and `target` columns; `target` must contain the two classes described by the notebook (`0` = non-disaster, `1` = disaster).

```powershell
python -m pip install -r requirements.txt
python train_model.py "twitter_disaster (1) (1).xlsx"
python app.py
```

The training command saves the complete vectorizer and classifier to `model.joblib`. The web interface is available at `http://localhost:5000`.

## Deploy

For Render, connect the repository containing this directory. The included `render.yaml` installs dependencies, trains the model during the build, starts Gunicorn, and configures `/healthz` as the health check. Ensure the labeled dataset is included at `TRAINING_DATA_PATH` (or change that setting in `render.yaml`).

For another host that supports a `Procfile`, set the build command to install dependencies and train the model, for example:

```sh
pip install -r requirements.txt && python train_model.py "twitter_disaster (1) (1).xlsx"
```

The dataset must be available to the build environment at that path. Do not commit it unless its license and privacy terms allow that. Set the service start command to `gunicorn app:app`; the included `Procfile` provides the same command. The host must provide its assigned port through `PORT`.

## Endpoints

- `GET /` serves the tweet classification form.
- `POST /api/predict` accepts `{"tweet": "..."}` and returns the predicted class, numeric label, and model confidence.
- `GET /healthz` reports whether the model artifact loaded.

The model artifact is intentionally generated during training rather than committed. Keep the training data and the model artifact from the same training run.