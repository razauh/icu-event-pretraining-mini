EXPERIMENT_IDS = (
    "EXP-00",
    "EXP-01",
    "EXP-02",
    "EXP-03",
    "EXP-04",
    "EXP-05"
)

EXPERIMENT_REGISTRY = {
    "EXP-00": {
        "representation": "timegap_static",
        "model_type": "logistic_regression",
        "pretraining": False,
        "evaluation": "patient_grouped_test"
    },
    "EXP-01": {
        "representation": "timegap_static",
        "model_type": "icu_tiny_transformer",
        "pretraining": False,
        "evaluation": "patient_grouped_test"
    },
    "EXP-02": {
        "representation": "timegap_static",
        "model_type": "icu_tiny_transformer",
        "pretraining": True,
        "evaluation": "patient_grouped_test"
    },
    "EXP-03": {
        "representation": "basic",
        "model_type": "icu_tiny_transformer",
        "pretraining": True,
        "evaluation": "patient_grouped_test"
    },
    "EXP-04": {
        "representation": "timegap_static",
        "model_type": "selected_model",
        "pretraining": "as_selected",
        "evaluation": "hospital_grouped_cv"
    },
    "EXP-05": {
        "representation": "timegap_static",
        "model_type": "icu_tiny_transformer",
        "pretraining": True,
        "evaluation": "hospital_cluster_fedavg"
    }
}
