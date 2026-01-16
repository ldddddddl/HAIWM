import wandb

api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("lddddl/HAIWM/w96f1w52")

# save the metrics for the run to a csv file
metrics_dataframe = run.history(samples=10_000_000, pandas=True)
metrics_dataframe.to_csv("metrics.csv")
