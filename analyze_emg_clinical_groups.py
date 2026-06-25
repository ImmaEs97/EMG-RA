"""
Analyze EMG features against clinical joint assessment data (CDAI).

The script expects:
- an EMG CSV with one or more numeric feature columns plus patient_id and exercise_id;
- a clinical CSV with patient_id, joint_group, right_tender_x_count, and
  right_swollen_x_count.

It produces:
- Spearman correlations between EMG features and clinical joint involvement rates;
- FDR-adjusted p-values for the continuous correlation analysis;
- an optional two-group Mann-Whitney comparison when --case-patients is provided;
- CSV summaries and, for the group comparison, a normalized feature boxplot.

The script is dataset-agnostic: patient groups, joint groups, labels, plot features,
and output paths can be configured from the command line.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests


METADATA_COLUMNS = {"patient_id", "exercise_id", "joint_group", "file_name"}
DEFAULT_JOINT_GROUPS = ["WRIST_RIGHT", "PIP_RIGHT", "MCP_RIGHT"]
DEFAULT_PLOT_FEATURES = ["rms", "iemg", "mav", "wl"]


# ---------------------------------------------------------------------------
# Command-line parsing helpers
# ---------------------------------------------------------------------------

def parse_comma_separated(value):
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_patient_ids(value):
    ids = parse_comma_separated(value)
    parsed_ids = []
    for patient_id in ids:
        try:
            parsed_ids.append(int(patient_id))
        except ValueError:
            parsed_ids.append(patient_id)
    return parsed_ids


# ---------------------------------------------------------------------------
# Shared data utilities
# ---------------------------------------------------------------------------

def get_feature_columns(df):
    """Return numeric EMG feature columns, excluding known metadata columns."""
    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def adjust_pvalues(df, pvalue_column, output_column, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction while preserving NaN p-values."""
    adjusted = np.full(len(df), np.nan)
    valid_mask = df[pvalue_column].notna().to_numpy()

    if valid_mask.any():
        _, valid_adjusted, _, _ = multipletests(
            df.loc[valid_mask, pvalue_column],
            alpha=alpha,
            method="fdr_bh",
        )
        adjusted[valid_mask] = valid_adjusted

    df[output_column] = adjusted
    return df


# ---------------------------------------------------------------------------
# Continuous analysis: EMG features vs clinical involvement rate
# ---------------------------------------------------------------------------

def compute_correlations(emg, clinical, feature_columns, joint_groups, label, count_column):
    """
    Compute Spearman correlations for each exercise, joint group, and EMG feature.

    For each patient, the EMG value is averaged within an exercise. The clinical
    value is the rate of involved joints within the requested joint group.
    """
    results = []

    for exercise_id in sorted(emg["exercise_id"].dropna().unique()):
        exercise_emg = emg[emg["exercise_id"] == exercise_id]

        for joint_group in joint_groups:
            for feature in feature_columns:
                pairs = []

                for patient_id in exercise_emg["patient_id"].dropna().unique():
                    emg_value = exercise_emg.loc[
                        exercise_emg["patient_id"] == patient_id,
                        feature,
                    ].mean()

                    clinical_mask = (
                        (clinical["patient_id"] == patient_id)
                        & (clinical["joint_group"] == joint_group)
                    )
                    joint_count = clinical_mask.sum()
                    if joint_count == 0:
                        continue

                    if count_column == "tender_or_swollen":
                        involved_count = (
                            clinical.loc[clinical_mask, "right_tender_x_count"].sum()
                            + clinical.loc[clinical_mask, "right_swollen_x_count"].sum()
                        )
                    else:
                        involved_count = clinical.loc[clinical_mask, count_column].sum()

                    clinical_rate = involved_count / joint_count
                    if not np.isnan(emg_value) and not np.isnan(clinical_rate):
                        pairs.append((emg_value, clinical_rate))

                if len(pairs) > 2:
                    emg_values, clinical_rates = zip(*pairs)
                    correlation, pvalue = spearmanr(emg_values, clinical_rates)
                else:
                    correlation, pvalue = np.nan, np.nan

                results.append(
                    {
                        "exercise_id": exercise_id,
                        "joint_group": joint_group,
                        "feature": feature,
                        "label": label,
                        "n_pairs": len(pairs),
                        "spearman_corr": correlation,
                        "spearman_p": pvalue,
                    }
                )

    return pd.DataFrame(results)


def run_continuous_analysis(emg, clinical, feature_columns, joint_groups, out_dir):
    """Run continuous Tender/Swollen/TenderOrSwollen correlation analyses."""
    analyses = [
        ("TenderOrSwollen", "tender_or_swollen"),
        ("Tender", "right_tender_x_count"),
        ("Swollen", "right_swollen_x_count"),
    ]
    summary_lines = []

    for label, count_column in analyses:
        correlation_df = compute_correlations(
            emg,
            clinical,
            feature_columns,
            joint_groups,
            label,
            count_column,
        )

        if not correlation_df.empty:
            correlation_df = adjust_pvalues(
                correlation_df,
                "spearman_p",
                "spearman_p_adj_05",
                alpha=0.05,
            )
            correlation_df["significant_adj_05"] = (
                (correlation_df["spearman_p_adj_05"] < 0.05)
                & (correlation_df["spearman_corr"].abs() > 0.5)
            )
            correlation_df = adjust_pvalues(
                correlation_df,
                "spearman_p",
                "spearman_p_adj_10",
                alpha=0.10,
            )
            correlation_df["significant_adj_10"] = (
                (correlation_df["spearman_p_adj_10"] < 0.10)
                & (correlation_df["spearman_corr"].abs() > 0.5)
            )

        output_path = out_dir / f"emg_clinical_continuous_corr_{label}.csv"
        correlation_df.to_csv(output_path, index=False)
        print(f"Continuous analysis saved to {output_path}")

        summary_lines.extend(
            build_continuous_summary(label, correlation_df)
        )

    summary_path = out_dir / "emg_clinical_continuous_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Continuous analysis summary saved to {summary_path}")


def build_continuous_summary(label, correlation_df):
    """Create a human-readable text summary for one continuous analysis."""
    summary_lines = [
        f"CONTINUOUS ANALYSIS RESULTS ({label})",
        (
            "Spearman correlation between EMG features and the clinical joint "
            "involvement rate for each exercise and joint group."
        ),
    ]

    if correlation_df.empty:
        summary_lines.append("No valid correlations were computed.\n")
        return summary_lines

    for alpha_label, significant_column, pvalue_column in [
        ("0.05", "significant_adj_05", "spearman_p_adj_05"),
        ("0.10", "significant_adj_10", "spearman_p_adj_10"),
    ]:
        significant_df = correlation_df[correlation_df[significant_column]]
        summary_lines.append(
            (
                f"FDR alpha={alpha_label}: {len(significant_df)} of "
                f"{len(correlation_df)} correlations are strong and significant "
                f"(|corr| > 0.5 and adjusted p < {alpha_label})."
            )
        )

        if significant_df.empty:
            summary_lines.append("No significant correlations found.")
        else:
            summary_lines.append(f"Significant combinations at FDR alpha={alpha_label}:")
            for _, row in significant_df.iterrows():
                direction = "positive" if row["spearman_corr"] > 0 else "negative"
                summary_lines.append(
                    (
                        f"  exercise {row['exercise_id']}, joint {row['joint_group']}, "
                        f"feature {row['feature']}, corr={row['spearman_corr']:.2f} "
                        f"({direction}), adjusted p={row[pvalue_column]:.3g}"
                    )
                )

    summary_lines.append("")
    return summary_lines


# ---------------------------------------------------------------------------
# Optional two-group analysis: Mann-Whitney comparison and boxplot
# ---------------------------------------------------------------------------

def run_group_comparison(
    emg,
    feature_columns,
    case_patients,
    case_label,
    control_label,
    plot_features,
    out_dir,
    show_plots=False,
):
    """
    Compare EMG features between a user-defined case group and all other patients.

    The case group is provided through --case-patients. If it is not provided,
    this whole block is skipped.
    """
    if not case_patients:
        print("Group comparison skipped: pass --case-patients to enable it.")
        return

    case_patients = set(case_patients)
    all_patients = set(emg["patient_id"].dropna().unique())
    control_patients = all_patients - case_patients

    results = []
    for exercise_id in sorted(emg["exercise_id"].dropna().unique()):
        exercise_emg = emg[emg["exercise_id"] == exercise_id]

        for feature in feature_columns:
            case_values = exercise_emg.loc[
                exercise_emg["patient_id"].isin(case_patients),
                feature,
            ].dropna()
            control_values = exercise_emg.loc[
                exercise_emg["patient_id"].isin(control_patients),
                feature,
            ].dropna()

            if len(case_values) > 0 and len(control_values) > 0:
                test_result = mannwhitneyu(
                    case_values,
                    control_values,
                    alternative="two-sided",
                )
                u_statistic = test_result.statistic
                pvalue = test_result.pvalue
            else:
                u_statistic = np.nan
                pvalue = np.nan

            results.append(
                {
                    "exercise_id": exercise_id,
                    "feature": feature,
                    "n_case": len(case_values),
                    "n_control": len(control_values),
                    "mean_case": case_values.mean(),
                    "mean_control": control_values.mean(),
                    "mannwhitney_u": u_statistic,
                    "mannwhitney_p": pvalue,
                }
            )

    group_df = pd.DataFrame(results)
    if not group_df.empty:
        group_df = adjust_pvalues(
            group_df,
            "mannwhitney_p",
            "mannwhitney_p_adj",
            alpha=0.05,
        )
        group_df["significant_adj"] = group_df["mannwhitney_p_adj"] < 0.05

    output_path = out_dir / "emg_group_comparison.csv"
    group_df.to_csv(output_path, index=False)
    print(f"Group comparison saved to {output_path}")

    write_group_summary(group_df, case_label, control_label, out_dir)
    plot_group_features(
        emg,
        case_patients,
        plot_features,
        case_label,
        control_label,
        out_dir,
        show_plots=show_plots,
    )


def write_group_summary(group_df, case_label, control_label, out_dir):
    """Write a compact text summary of the Mann-Whitney group comparison."""
    significant_df = group_df[group_df["significant_adj"]] if not group_df.empty else group_df
    summary_lines = [
        f"EMG group comparison ({case_label} vs {control_label})",
        (
            f"{len(significant_df)} of {len(group_df)} comparisons are significant "
            "after FDR correction (adjusted p < 0.05)."
        ),
    ]

    if significant_df.empty:
        summary_lines.append("No significant group differences were found.")
    else:
        summary_lines.append("Significant combinations:")
        for _, row in significant_df.iterrows():
            summary_lines.append(
                (
                    f"  exercise {row['exercise_id']}, feature {row['feature']}, "
                    f"mean_case={row['mean_case']:.2f}, "
                    f"mean_control={row['mean_control']:.2f}, "
                    f"adjusted p={row['mannwhitney_p_adj']:.3g}"
                )
            )

    summary_path = out_dir / "emg_group_comparison_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Group comparison summary saved to {summary_path}")


def plot_group_features(
    emg,
    case_patients,
    plot_features,
    case_label,
    control_label,
    out_dir,
    show_plots=False,
):
    """Save a z-score normalized boxplot for selected EMG features by group."""
    available_features = [
        feature for feature in plot_features
        if feature in emg.columns and pd.api.types.is_numeric_dtype(emg[feature])
    ]
    if not available_features:
        print("Feature plot skipped: none of the requested plot features are available.")
        return

    plot_df = emg.copy()
    plot_df["group"] = np.where(
        plot_df["patient_id"].isin(case_patients),
        case_label,
        control_label,
    )

    normalized_rows = []
    for feature in available_features:
        mean = plot_df[feature].mean()
        std = plot_df[feature].std()
        if pd.isna(std) or std == 0:
            continue

        normalized_values = (plot_df[feature] - mean) / std
        normalized_rows.extend(
            {
                "Feature": feature,
                "Value": value,
                "Group": group,
            }
            for value, group in zip(normalized_values, plot_df["group"])
            if not pd.isna(value)
        )

    if not normalized_rows:
        print("Feature plot skipped: selected features cannot be normalized.")
        return

    normalized_df = pd.DataFrame(normalized_rows)
    plt.figure(figsize=(12, 6))
    axis = sns.boxplot(
        x="Feature",
        y="Value",
        hue="Group",
        data=normalized_df,
        palette="Set2",
    )
    plt.title("Feature distribution by group (z-score normalized)", fontsize=18)
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Normalized value (z-score)", fontsize=14)
    plt.legend(title="Group", fontsize=12, title_fontsize=12)
    axis.tick_params(axis="x", labelsize=12)
    axis.tick_params(axis="y", labelsize=12)
    plt.tight_layout()

    output_path = out_dir / "emg_group_comparison_boxplot.png"
    plt.savefig(output_path, dpi=150)
    print(f"Group comparison plot saved to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Input validation and entry point
# ---------------------------------------------------------------------------

def validate_input_columns(emg, clinical):
    """Fail early if the input CSV files do not contain the required columns."""
    required_emg_columns = {"patient_id", "exercise_id"}
    required_clinical_columns = {
        "patient_id",
        "joint_group",
        "right_tender_x_count",
        "right_swollen_x_count",
    }

    missing_emg = required_emg_columns - set(emg.columns)
    missing_clinical = required_clinical_columns - set(clinical.columns)

    if missing_emg:
        raise ValueError(f"Missing required EMG columns: {sorted(missing_emg)}")
    if missing_clinical:
        raise ValueError(f"Missing required clinical columns: {sorted(missing_clinical)}")


def parse_args():
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze EMG features against clinical joint involvement data. "
            "The script computes continuous Spearman correlations and, when "
            "requested, a two-group Mann-Whitney comparison."
        )
    )
    parser.add_argument("--emg-csv", required=True, help="Input EMG feature CSV file.")
    parser.add_argument(
        "--clinical-csv",
        required=True,
        help="Input clinical joint assessment CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory where result files will be written.",
    )
    parser.add_argument(
        "--joint-groups",
        default=",".join(DEFAULT_JOINT_GROUPS),
        help="Comma-separated clinical joint groups to analyze.",
    )
    parser.add_argument(
        "--case-patients",
        default="",
        help=(
            "Comma-separated patient IDs for the first comparison group. "
            "If omitted, the two-group comparison is skipped."
        ),
    )
    parser.add_argument(
        "--case-label",
        default="Case",
        help="Display label for patients listed in --case-patients.",
    )
    parser.add_argument(
        "--control-label",
        default="Control",
        help="Display label for all remaining patients.",
    )
    parser.add_argument(
        "--plot-features",
        default=",".join(DEFAULT_PLOT_FEATURES),
        help="Comma-separated EMG features to include in the group boxplot.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def main():
    """Load input data, validate it, then run the requested analyses."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emg = pd.read_csv(args.emg_csv)
    clinical = pd.read_csv(args.clinical_csv)
    validate_input_columns(emg, clinical)

    feature_columns = get_feature_columns(emg)
    if not feature_columns:
        raise ValueError("No numeric EMG feature columns were found.")

    joint_groups = parse_comma_separated(args.joint_groups)
    case_patients = parse_patient_ids(args.case_patients)
    plot_features = parse_comma_separated(args.plot_features)

    print(f"EMG features used: {feature_columns}")
    print(f"Joint groups used: {joint_groups}")

    run_continuous_analysis(
        emg,
        clinical,
        feature_columns,
        joint_groups,
        out_dir,
    )
    run_group_comparison(
        emg,
        feature_columns,
        case_patients,
        args.case_label,
        args.control_label,
        plot_features,
        out_dir,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
