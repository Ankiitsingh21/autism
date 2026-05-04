import pandas as pd
import numpy as np
import warnings
import time
import os

warnings.filterwarnings("ignore")

# ── Rich Terminal UI ──────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.rule import Rule
from rich.align import Align

# ── ML Libraries ─────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")           # headless – saves to PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
console = Console()
DATASET_PATH = "asd_dataset.csv"
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════════
def show_banner():
    banner = Text()
    banner.append("\n  ASD DETECTION  ", style="bold white on blue")
    banner.append("  ML Pipeline  ", style="bold white on dark_green")
    console.print(Align.center(banner))
    console.print(Align.center(
        Text("Logistic Regression · SMOTE · SHAP · Cross-Validation", style="dim")
    ))
    console.print()

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════
def load_data():
    console.print(Rule("[bold cyan]STEP 1 — Loading Dataset[/bold cyan]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Reading CSV file...", total=3)

        data = pd.read_csv(DATASET_PATH)
        progress.advance(t)
        time.sleep(0.3)

        data.columns = data.columns.str.strip().str.replace("'", "", regex=False)
        progress.advance(t)
        time.sleep(0.2)

        data = data.replace("?", np.nan).dropna().reset_index(drop=True)
        progress.advance(t)
        time.sleep(0.2)

    # Summary table
    tbl = Table(title="Dataset Overview", box=box.ROUNDED, border_style="cyan", show_lines=True)
    tbl.add_column("Property", style="bold yellow")
    tbl.add_column("Value", style="green")
    tbl.add_row("Total Rows",    str(len(data)))
    tbl.add_row("Total Columns", str(len(data.columns)))
    tbl.add_row("Missing Values (after clean)", "0")
    tbl.add_row("File", DATASET_PATH)
    console.print(tbl)
    console.print()

    return data

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESS
# ═══════════════════════════════════════════════════════════════════
def preprocess(data):
    console.print(Rule("[bold cyan]STEP 2 — Preprocessing[/bold cyan]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Encoding features...", total=5)

        # Drop unused
        for col in ["Case_No", "Who completed the test"]:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
        progress.advance(t); time.sleep(0.2)

        # Target
        target_col = next(c for c in data.columns if "ASD" in c.upper())
        y = data[target_col].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
        X = data.drop(columns=[target_col])
        progress.advance(t); time.sleep(0.2)

        # YES/NO → 1/0
        X = X.replace({"yes": 1, "no": 0, "YES": 1, "NO": 0, "Yes": 1, "No": 0})
        progress.advance(t); time.sleep(0.2)

        # One-hot
        X = pd.get_dummies(X, drop_first=False)
        progress.advance(t); time.sleep(0.2)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        progress.advance(t); time.sleep(0.2)

    # Class distribution table
    dist = y.value_counts()
    tbl = Table(title="Class Distribution", box=box.ROUNDED, border_style="cyan", show_lines=True)
    tbl.add_column("Class",   style="bold yellow")
    tbl.add_column("Label",   style="white")
    tbl.add_column("Count",   style="green")
    tbl.add_column("Share",   style="blue")
    for val, label in [(1, "ASD Positive"), (0, "ASD Negative")]:
        tbl.add_row(str(val), label, str(dist[val]), f"{dist[val]/len(y)*100:.1f}%")
    console.print(tbl)
    console.print()

    return X, y, X_train, X_test, y_train, y_test, target_col

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — SMOTE + SCALE
# ═══════════════════════════════════════════════════════════════════
def balance_and_scale(X_train, X_test, y_train):
    console.print(Rule("[bold cyan]STEP 3 — SMOTE Balancing & Scaling[/bold cyan]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Applying SMOTE...", total=2)

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        progress.advance(t); time.sleep(0.3)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_res)
        X_test_sc  = scaler.transform(X_test)
        progress.advance(t); time.sleep(0.2)

    after = pd.Series(y_train_res).value_counts()
    tbl = Table(title="After SMOTE (Train Set)", box=box.ROUNDED, border_style="cyan", show_lines=True)
    tbl.add_column("Class",  style="bold yellow")
    tbl.add_column("Count",  style="green")
    tbl.add_row("ASD Positive (1)", str(after[1]))
    tbl.add_row("ASD Negative (0)", str(after[0]))
    console.print(tbl)
    console.print()

    return X_train_sc, X_test_sc, y_train_res, scaler

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════
def train_model(X_train_sc, y_train_res):
    console.print(Rule("[bold cyan]STEP 4 — Model Training[/bold cyan]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Training Logistic Regression...", total=1)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_sc, y_train_res)
        progress.advance(t); time.sleep(0.5)

    console.print("[green]✔[/green]  Model trained successfully\n")
    return model

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE
# ═══════════════════════════════════════════════════════════════════
def evaluate(model, X_train_sc, y_train_res, X_test_sc, y_test, feature_names):
    console.print(Rule("[bold cyan]STEP 5 — Evaluation[/bold cyan]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Computing metrics...", total=3)

        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        progress.advance(t); time.sleep(0.2)

        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_proba)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cv_scores = cross_val_score(model, X_train_sc, y_train_res, cv=5, scoring="accuracy")
        progress.advance(t); time.sleep(0.3)

        # Feature importance
        importance = np.abs(model.coef_[0])
        feat_df = pd.DataFrame({"feature": feature_names, "importance": importance})\
                    .sort_values("importance", ascending=False)
        progress.advance(t); time.sleep(0.1)

    # ── Metric panels ──────────────────────────────────────────
    def metric_panel(label, value, color="green"):
        return Panel(
            Align.center(Text(value, style=f"bold {color} on black")),
            title=f"[bold]{label}[/bold]",
            border_style=color,
            width=22,
        )

    console.print(Columns([
        metric_panel("Accuracy",   f"{acc*100:.2f}%",       "green"),
        metric_panel("ROC-AUC",    f"{auc:.4f}",            "cyan"),
        metric_panel("CV Acc (5F)",f"{cv_scores.mean():.4f}","yellow"),
        metric_panel("CV Std",     f"± {cv_scores.std():.4f}","magenta"),
    ]))
    console.print()

    # ── Classification report ──────────────────────────────────
    cr_tbl = Table(title="Classification Report", box=box.ROUNDED, border_style="cyan", show_lines=True)
    for col in ["Class", "Precision", "Recall", "F1-Score", "Support"]:
        cr_tbl.add_column(col, style="white", justify="center")
    for label, name in [("0", "ASD Negative"), ("1", "ASD Positive")]:
        r = report[label]
        cr_tbl.add_row(
            name,
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1-score']:.3f}",
            str(int(r["support"])),
        )
    console.print(cr_tbl)
    console.print()

    # ── Confusion matrix (text) ────────────────────────────────
    tn, fp, fn, tp = cm.ravel()
    cm_tbl = Table(title="Confusion Matrix", box=box.ROUNDED, border_style="magenta", show_lines=True)
    cm_tbl.add_column("",             style="bold dim")
    cm_tbl.add_column("Pred Neg (0)", style="cyan",  justify="center")
    cm_tbl.add_column("Pred Pos (1)", style="yellow", justify="center")
    cm_tbl.add_row("Actual Neg (0)", str(tn), f"[red]{fp}[/red]")
    cm_tbl.add_row("Actual Pos (1)", f"[red]{fn}[/red]", str(tp))
    console.print(cm_tbl)
    console.print()

    return y_pred, y_proba, cm, feat_df, acc, auc, cv_scores

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — GRAPHS
# ═══════════════════════════════════════════════════════════════════
def generate_graphs(model, X_test_sc, y_test, y_proba, cm, feat_df, feature_names, X):
    console.print(Rule("[bold cyan]STEP 6 — Generating Plots[/bold cyan]"))

    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#c9d1d9",
        "xtick.color":      "#c9d1d9",
        "ytick.color":      "#c9d1d9",
        "text.color":       "#c9d1d9",
        "grid.color":       "#21262d",
        "font.family":      "monospace",
    })

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Rendering plots...", total=4)

        # ── 1. Dashboard (confusion + roc + feature importance) ──
        fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
        fig.suptitle("ASD Detection — Model Dashboard", fontsize=18,
                     color="#58a6ff", fontweight="bold", y=0.98)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        ACCENT   = "#58a6ff"
        GREEN    = "#3fb950"
        RED      = "#f85149"
        PURPLE   = "#bc8cff"
        ORANGE   = "#d29922"

        # Confusion Matrix heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                    linewidths=2, linecolor="#0d1117",
                    annot_kws={"size": 16, "weight": "bold", "color": "white"})
        ax1.set_title("Confusion Matrix", color=ACCENT, fontsize=13, pad=10)
        ax1.set_xlabel("Predicted", color="#c9d1d9")
        ax1.set_ylabel("Actual",    color="#c9d1d9")

        # ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
        ax2.plot([0,1],[0,1], "--", color="#484f58", lw=1.5, label="Random")
        ax2.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
        ax2.set_title("ROC Curve", color=ACCENT, fontsize=13, pad=10)
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.legend(framealpha=0.2, labelcolor="white")
        ax2.set_facecolor("#161b22")

        # Top 10 Feature Importance
        ax3 = fig.add_subplot(gs[0, 2])
        top10 = feat_df.head(10)
        colors = [ACCENT if i < 3 else PURPLE if i < 7 else GREEN for i in range(10)]
        bars = ax3.barh(top10["feature"][::-1], top10["importance"][::-1],
                        color=colors[::-1], edgecolor="#0d1117", linewidth=0.5)
        ax3.set_title("Top 10 Features (|Coef|)", color=ACCENT, fontsize=13, pad=10)
        ax3.set_xlabel("Importance")
        for bar, val in zip(bars, top10["importance"][::-1]):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=8, color="#c9d1d9")
        ax3.set_facecolor("#161b22")

        # Class distribution pie
        ax4 = fig.add_subplot(gs[1, 0])
        dist = [sum(y_test == 0), sum(y_test == 1)]
        ax4.pie(dist, labels=["ASD Neg", "ASD Pos"],
                colors=[GREEN, RED], autopct="%1.1f%%",
                startangle=90, textprops={"color": "white", "fontsize": 11},
                wedgeprops={"linewidth": 2, "edgecolor": "#0d1117"})
        ax4.set_title("Test Set Distribution", color=ACCENT, fontsize=13, pad=10)
        ax4.set_facecolor("#161b22")

        # Score distributions
        ax5 = fig.add_subplot(gs[1, 1])
        score_cols = [c for c in X.columns if "Score" in c]
        score_means = X[score_cols].mean().sort_values(ascending=False)
        ax5.bar(score_means.index, score_means.values, color=ORANGE, edgecolor="#0d1117")
        ax5.set_title("Avg AQ Score per Question", color=ACCENT, fontsize=13, pad=10)
        ax5.set_xlabel("Question"); ax5.set_ylabel("Mean Score")
        ax5.set_xticklabels(score_means.index, rotation=45, ha="right", fontsize=9)
        ax5.set_facecolor("#161b22")

        # Predicted probability histogram
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist([y_proba[y_test==0], y_proba[y_test==1]],
                 bins=20, color=[GREEN, RED], label=["ASD Neg", "ASD Pos"],
                 edgecolor="#0d1117", alpha=0.8)
        ax6.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold 0.5")
        ax6.set_title("Predicted Probability Dist.", color=ACCENT, fontsize=13, pad=10)
        ax6.set_xlabel("P(ASD Positive)"); ax6.set_ylabel("Count")
        ax6.legend(framealpha=0.2, labelcolor="white", fontsize=9)
        ax6.set_facecolor("#161b22")

        dashboard_path = f"{OUTPUT_DIR}/dashboard.png"
        fig.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        progress.advance(t); time.sleep(0.2)

        # ── 2. SHAP ──────────────────────────────────────────────
        shap_path = None
        if SHAP_AVAILABLE:
            fn_list = list(feature_names)
            X_tr_df  = pd.DataFrame(X_test_sc, columns=fn_list)
            explainer = shap.LinearExplainer(model, X_tr_df)
            sv = explainer.shap_values(X_tr_df)
            if isinstance(sv, list): sv = sv[1]

            fig2, (sa, sb) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0d1117")
            fig2.suptitle("SHAP Feature Importance", fontsize=16, color="#58a6ff", fontweight="bold")

            # Manual SHAP bar (works headless)
            mean_abs = np.abs(sv).mean(axis=0)
            idx = np.argsort(mean_abs)[-15:]
            feat_labels = np.array(fn_list)[idx]
            bar_colors = ["#58a6ff" if v > 0 else "#f85149" for v in mean_abs[idx]]
            sb.barh(feat_labels, mean_abs[idx], color=bar_colors, edgecolor="#0d1117")
            sb.set_title("SHAP |Mean| — Top 15", color="#58a6ff", fontsize=13)
            sb.set_xlabel("Mean |SHAP value|"); sb.set_facecolor("#161b22")

            # Scatter: first test sample SHAP values
            sa.barh(feat_labels, sv[0][idx], color=["#3fb950" if v>=0 else "#f85149" for v in sv[0][idx]],
                    edgecolor="#0d1117")
            sa.axvline(0, color="white", lw=1)
            sa.set_title("SHAP — First Sample Breakdown", color="#58a6ff", fontsize=13)
            sa.set_xlabel("SHAP value"); sa.set_facecolor("#161b22")
            for ax in [sa, sb]:
                ax.tick_params(colors="#c9d1d9"); ax.set_facecolor("#161b22")

            shap_path = f"{OUTPUT_DIR}/shap_analysis.png"
            fig2.savefig(shap_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig2)
        progress.advance(t); time.sleep(0.2)

        # ── 3. Correlation heatmap for AQ scores ─────────────────
        score_data = X[[c for c in X.columns if "Score" in c]].copy()
        fig3, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        corr = score_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", ax=ax, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", linewidths=0.5, linecolor="#0d1117",
                    annot_kws={"size": 9})
        ax.set_title("AQ Score Correlation Matrix", color="#58a6ff", fontsize=14, pad=12)
        corr_path = f"{OUTPUT_DIR}/correlation.png"
        fig3.savefig(corr_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig3)
        progress.advance(t); time.sleep(0.2)

        progress.advance(t)

    # Report saved paths
    saved = [
        ("Dashboard",    dashboard_path),
        ("Correlation",  corr_path),
    ]
    if shap_path:
        saved.append(("SHAP Analysis", shap_path))

    tbl = Table(title="Saved Plots", box=box.ROUNDED, border_style="green", show_lines=True)
    tbl.add_column("Plot",  style="bold yellow")
    tbl.add_column("Path",  style="cyan")
    for name, path in saved:
        tbl.add_row(name, path)
    console.print(tbl)
    console.print()

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    show_banner()

    data                                              = load_data()
    X, y, X_train, X_test, y_train, y_test, tgt_col = preprocess(data)
    X_train_sc, X_test_sc, y_train_res, scaler       = balance_and_scale(X_train, X_test, y_train)
    model                                             = train_model(X_train_sc, y_train_res)
    y_pred, y_proba, cm, feat_df, acc, auc, cv_sc    = evaluate(
        model, X_train_sc, y_train_res, X_test_sc, y_test, X.columns.tolist()
    )
    generate_graphs(model, X_test_sc, y_test, y_proba, cm,
                    feat_df, X.columns.tolist(), X)

    # Final summary panel
    summary = (
        f"[bold green]Accuracy :[/bold green] {acc*100:.2f}%\n"
        f"[bold cyan]ROC-AUC  :[/bold cyan] {auc:.4f}\n"
        f"[bold yellow]CV Acc   :[/bold yellow] {cv_sc.mean():.4f} ± {cv_sc.std():.4f}\n"
        f"[bold white]Plots    :[/bold white] saved to [cyan]{OUTPUT_DIR}/[/cyan]"
    )
    console.print(Panel(summary, title="[bold white]Pipeline Complete ✔[/bold white]",
                        border_style="green", padding=(1, 4)))

if __name__ == "__main__":
    main()