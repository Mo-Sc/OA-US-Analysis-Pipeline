from scipy.stats import f_oneway, kruskal, spearmanr
from scipy.stats import ttest_ind


def compute_stat_test(stat_test, df, group_data):
    """
    Can be used to calculate selected statistical tests
    """
    if stat_test == "anova":
        stat, p = f_oneway(*group_data)
        return f"ANOVA p = {p:.4f}"
    elif stat_test == "kruskal":
        stat, p = kruskal(*group_data)
        return f"Kruskal p = {p:.4f}"
    elif stat_test == "spearman":
        rho, p = spearmanr(df["GroupIndex"], df["Mean Intensity"])
        return f"Spearman ρ = {rho:.2f}, p = {p:.4f}"
    elif stat_test == "ttest_student":
        assert len(group_data) == 2, "Student's T-test requires exactly two groups."
        stat, p = ttest_ind(*group_data, equal_var=True)
        return f"Student's T-test p = {p:.4f}"
    elif stat_test == "ttest_welch":
        assert len(group_data) == 2, "Welch's T-test requires exactly two groups."
        stat, p = ttest_ind(*group_data, equal_var=False)
        return f"Welch's T-test p = {p:.4f}"
    return None
