"""
Log DistilBERT training experiment to MLflow.
Run once after the trained model is placed in BERT_DATASET/bert_model/.
"""

import os
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI    = "http://localhost:5001/"
MODEL_DIR     = "/home/ubuntu/Mlflow/BERT_DATASET/bert_model"
METRICS_FILE  = os.path.join(MODEL_DIR, "training_metrics.json")
CM_IMAGE      = os.path.join(MODEL_DIR, "confusion_matrix_bert.png")

def main():
    if not os.path.exists(METRICS_FILE):
        print(f"ERROR: {METRICS_FILE} not found. Place training_metrics.json in bert_model/.")
        sys.exit(1)

    with open(METRICS_FILE) as f:
        history = json.load(f)

    final = history[-1]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("bert-sentiment")

    with mlflow.start_run(run_name="distilbert-base-uncased-v1") as run:
        # ── Hyperparameters ───────────────────────────────────────────────────
        mlflow.log_params({
            "model_name":       "distilbert-base-uncased",
            "num_labels":       3,
            "max_length":       128,
            "epochs":           len(history),
            "batch_size":       16,
            "learning_rate":    "2e-5",
            "weight_decay":     0.01,
            "gradient_accum":   2,
            "warmup_ratio":     0.1,
            "class_weighting":  True,
            "train_rows":       77120,
            "test_rows":        19281,
            "data_sources":     "Reddit + TweetEval",
            "label_schema":     "-1=negative, 0=neutral, 1=positive",
        })

        # ── Per-epoch metrics ─────────────────────────────────────────────────
        for entry in history:
            epoch = entry["epoch"]
            mlflow.log_metrics({
                "train_loss":     round(entry["loss"], 4),
                "test_accuracy":  round(entry["accuracy"], 4),
                "test_macro_f1":  round(entry["f1"], 4),
            }, step=epoch)

        # ── Final summary metrics ─────────────────────────────────────────────
        mlflow.log_metrics({
            "final_accuracy": round(final["accuracy"], 4),
            "final_macro_f1": round(final["f1"], 4),
            "final_loss":     round(final["loss"], 4),
        })

        # ── Artifacts ─────────────────────────────────────────────────────────
        mlflow.log_artifacts(MODEL_DIR, artifact_path="bert_model")
        if os.path.exists(CM_IMAGE):
            mlflow.log_artifact(CM_IMAGE, artifact_path="plots")

        # ── Register model ────────────────────────────────────────────────────
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/bert_model"
        client = MlflowClient()
        try:
            client.create_registered_model("bert-sentiment")
        except Exception:
            pass  # already exists
        mv = client.create_model_version(
            name="bert-sentiment",
            source=model_uri,
            run_id=run_id,
        )
        client.set_registered_model_alias("bert-sentiment", "champion", mv.version)

        print(f"\n✓ Run ID       : {run_id}")
        print(f"✓ Final Acc    : {final['accuracy']:.4f}")
        print(f"✓ Final F1     : {final['f1']:.4f}")
        print(f"✓ Model version: {mv.version}")
        print(f"✓ MLflow UI    : {MLFLOW_URI}")

if __name__ == "__main__":
    main()
