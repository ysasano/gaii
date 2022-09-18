from pathlib import Path
import fire
import pandas as pd
import visualize


def create_summary(experiment_dir):
    entropy_filename = f"{experiment_dir}/entropy.txt"
    df = pd.read_table(
        entropy_filename, names=("data", "partation", "model", "entropy")
    )
    data_list = df["data"].unique()

    for data_name in data_list:
        data_dir = Path(f"{experiment_dir}/summary/data={data_name}/")
        data_dir.mkdir(parents=True, exist_ok=True)
        df_per_data = df.query(f'data == "{data_name}"')
        df_pivot = df_per_data.pivot(
            index="partation", columns="model", values="entropy"
        )
        visualize.plot_result_all(df_pivot, data_dir)


if __name__ == "__main__":
    fire.Fire(create_summary)
