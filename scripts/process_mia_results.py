import fire
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.25)
cp = "hls"


def print_best(
    metric, prompts_of_samples, samples, success, name1, scores1, name2=None, scores2=None, n=10
):
    """
    print the `n` best samples according to the given `metric`
    """
    # these values then don't show up in the largest vals
    metric[np.isnan(metric)] = -np.inf
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(
                f"{i+1}: success={success[idx]}, {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
            )
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        pprint(f"PROMPT: {prompts_of_samples[idx]} --- TEXT: {samples[idx]}")
        print()
        print()


def process_mia_results(
    mia_results_file, plot_file=None, ratio_file=None, target_lm="target_ppl", ref_lm="ref_ppl"
):
    df = pd.read_csv(mia_results_file)
    n_empty_gens = len(df[df["gen"].isna()])
    df_gen = df.loc[df["gen"].notna()]
    success = df_gen["success"].sum()
    assert ref_lm in ["ref_ppl", "lower_ppl", "zlib"]

    # sort by ratio and optionally save
    metric = np.log(df_gen[ref_lm]) / np.log(df_gen[target_lm])
    print(f"======== top sample by ratio of {ref_lm} and {target_lm} perplexities: ========")
    print_best(
        metric.values,
        df_gen["prompt"].values,
        df_gen["gen"].values,
        df_gen["success"].values,
        "PPL-target_lm",
        df_gen[target_lm].values,
        "PPL-ref_lm",
        df_gen[ref_lm].values,
    )

    if ratio_file is not None:
        metric = df_gen[target_lm] / df_gen[ref_lm]
        metric[np.isnan(metric)] = -np.inf
        idxs = np.argsort(metric)
        df_gen.iloc[idxs].to_csv(ratio_file, index=False, header=True, sep=",")

    if plot_file is not None:
        fig = plt.figure()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = sns.scatterplot(
            data=df_gen,
            x=target_lm,
            y=ref_lm,
            hue="success",
            palette=sns.color_palette(cp, 2),
        )
        ax.set_yscale("log")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02))
        ax.set_title(f"MIA Attack PPL Ratio: success rate {success}, empty gens: {n_empty_gens}")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire(process_mia_results)
