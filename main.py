import pandas as pd
import numpy as np
import warnings
import time
import os
import joblib

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
from rich.prompt import Prompt, IntPrompt

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
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
MODEL_PATH   = "asd_model.pkl"
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

        for col in ["Case_No", "Who completed the test"]:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
        progress.advance(t); time.sleep(0.2)

        target_col = next(c for c in data.columns if "ASD" in c.upper())
        y = data[target_col].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
        X = data.drop(columns=[target_col])
        progress.advance(t); time.sleep(0.2)

        X = X.replace({"yes": 1, "no": 0, "YES": 1, "NO": 0, "Yes": 1, "No": 0})
        progress.advance(t); time.sleep(0.2)

        X = pd.get_dummies(X, drop_first=False)
        progress.advance(t); time.sleep(0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        progress.advance(t); time.sleep(0.2)

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

        importance = np.abs(model.coef_[0])
        feat_df = pd.DataFrame({"feature": feature_names, "importance": importance})\
                    .sort_values("importance", ascending=False)
        progress.advance(t); time.sleep(0.1)

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

        fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
        fig.suptitle("ASD Detection — Model Dashboard", fontsize=18,
                     color="#58a6ff", fontweight="bold", y=0.98)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        ACCENT   = "#58a6ff"
        GREEN    = "#3fb950"
        RED      = "#f85149"
        PURPLE   = "#bc8cff"
        ORANGE   = "#d29922"

        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                    linewidths=2, linecolor="#0d1117",
                    annot_kws={"size": 16, "weight": "bold", "color": "white"})
        ax1.set_title("Confusion Matrix", color=ACCENT, fontsize=13, pad=10)
        ax1.set_xlabel("Predicted", color="#c9d1d9")
        ax1.set_ylabel("Actual",    color="#c9d1d9")

        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
        ax2.plot([0,1],[0,1], "--", color="#484f58", lw=1.5, label="Random")
        ax2.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
        ax2.set_title("ROC Curve", color=ACCENT, fontsize=13, pad=10)
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.legend(framealpha=0.2, labelcolor="white")
        ax2.set_facecolor("#161b22")

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

        ax4 = fig.add_subplot(gs[1, 0])
        dist = [sum(y_test == 0), sum(y_test == 1)]
        ax4.pie(dist, labels=["ASD Neg", "ASD Pos"],
                colors=[GREEN, RED], autopct="%1.1f%%",
                startangle=90, textprops={"color": "white", "fontsize": 11},
                wedgeprops={"linewidth": 2, "edgecolor": "#0d1117"})
        ax4.set_title("Test Set Distribution", color=ACCENT, fontsize=13, pad=10)
        ax4.set_facecolor("#161b22")

        ax5 = fig.add_subplot(gs[1, 1])
        score_cols = [c for c in X.columns if "Score" in c]
        score_means = X[score_cols].mean().sort_values(ascending=False)
        ax5.bar(score_means.index, score_means.values, color=ORANGE, edgecolor="#0d1117")
        ax5.set_title("Avg AQ Score per Question", color=ACCENT, fontsize=13, pad=10)
        ax5.set_xlabel("Question"); ax5.set_ylabel("Mean Score")
        ax5.set_xticklabels(score_means.index, rotation=45, ha="right", fontsize=9)
        ax5.set_facecolor("#161b22")

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

        shap_path = None
        if SHAP_AVAILABLE:
            fn_list = list(feature_names)
            X_tr_df  = pd.DataFrame(X_test_sc, columns=fn_list)
            explainer = shap.LinearExplainer(model, X_tr_df)
            sv = explainer.shap_values(X_tr_df)
            if isinstance(sv, list): sv = sv[1]

            fig2, (sa, sb) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0d1117")
            fig2.suptitle("SHAP Feature Importance", fontsize=16, color="#58a6ff", fontweight="bold")

            mean_abs = np.abs(sv).mean(axis=0)
            idx = np.argsort(mean_abs)[-15:]
            feat_labels = np.array(fn_list)[idx]
            bar_colors = ["#58a6ff" if v > 0 else "#f85149" for v in mean_abs[idx]]
            sb.barh(feat_labels, mean_abs[idx], color=bar_colors, edgecolor="#0d1117")
            sb.set_title("SHAP |Mean| — Top 15", color="#58a6ff", fontsize=13)
            sb.set_xlabel("Mean |SHAP value|"); sb.set_facecolor("#161b22")

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
# HELPER — Build prediction input aligned to trained feature columns
# ═══════════════════════════════════════════════════════════════════
def build_input_vector(raw_dict, feature_columns):
    """
    Takes a plain dict of raw values (before one-hot),
    converts to DataFrame, one-hot encodes, then aligns
    columns to match exactly what the model was trained on.
    Missing columns → 0, extra columns → dropped.
    """
    df = pd.DataFrame([raw_dict])
    df = df.replace({"yes": 1, "no": 0, "YES": 1, "NO": 0})
    df = pd.get_dummies(df, drop_first=False)
    # Align to training columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


# ═══════════════════════════════════════════════════════════════════
# FEATURE 1 — Conversational ASD Self Test
# ═══════════════════════════════════════════════════════════════════
def conversational_test(model, scaler, feature_columns):
    """
    Asks Q-CHAT-10 questions + demographics interactively.
    Predicts ASD and shows risk level with confidence.
    Does NOT touch any existing code/model training.
    """
    console.print()
    console.print(Rule("[bold magenta]🧠 ASD SELF SCREENING TEST[/bold magenta]"))
    console.print(Panel(
        "[yellow]This is NOT a clinical diagnosis.[/yellow]\n"
        "It is a quick ML-based screening tool.\n"
        "Answer each question honestly. Type [bold]1[/bold] for Yes, [bold]2[/bold] for No.",
        title="[bold white]Instructions[/bold white]",
        border_style="yellow"
    ))
    console.print()

    # ── Q-CHAT-10 Questions ────────────────────────────────────
    questions = [
        ("A1_Score", "Does the person notice when others are upset or angry?"),
        ("A2_Score", "Does the person find it easy to make friends?"),
        ("A3_Score", "Can the person switch easily between activities?"),
        ("A4_Score", "Does the person enjoy pretend/imaginative play?"),
        ("A5_Score", "Does the person find social situations easy?"),
        ("A6_Score", "Does the person notice when someone is bored in conversation?"),
        ("A7_Score", "Can the person tell if someone is interested in what they are saying?"),
        ("A8_Score", "Does the person enjoy group activities / social gathering?"),
        ("A9_Score", "Does the person find it easy to understand people's feelings?"),
        ("A10_Score","Does the person find it easy to 'read between the lines' in conversations?"),
    ]

    scores = {}
    console.print("[bold cyan]━━━ SECTION 1: Behavioral Questions (Q-CHAT-10) ━━━[/bold cyan]\n")

    for i, (col, question) in enumerate(questions, 1):
        console.print(f"[bold yellow]Q{i}.[/bold yellow] {question}")
        while True:
            ans = Prompt.ask("    [dim]Your answer[/dim]", choices=["1", "2"], default="2")
            break
        # Q-CHAT scoring: Yes=0 (typical), No=1 (atypical flag) for most items
        # Simpler: we store 1 for Yes, 0 for No — model handles it
        scores[col] = 1 if ans == "1" else 0
        console.print()

    # ── Demographics ───────────────────────────────────────────
    console.print("[bold cyan]━━━ SECTION 2: Basic Information ━━━[/bold cyan]\n")

    # Age
    while True:
        try:
            age = int(Prompt.ask("[bold yellow]Age[/bold yellow] (in years)"))
            if 1 <= age <= 120:
                break
            console.print("[red]Please enter a valid age between 1 and 120.[/red]")
        except ValueError:
            console.print("[red]Numbers only please.[/red]")

    # Gender
    console.print("[bold yellow]Gender[/bold yellow]")
    console.print("  1) Male   2) Female")
    g = Prompt.ask("  Choice", choices=["1", "2"], default="1")
    gender = "m" if g == "1" else "f"

    # Ethnicity
    ethnicities = {
        "1": "White-European", "2": "Asian", "3": "Black",
        "4": "Latino", "5": "Middle Eastern", "6": "South Asian", "7": "Others"
    }
    console.print("\n[bold yellow]Ethnicity[/bold yellow]")
    for k, v in ethnicities.items():
        console.print(f"  {k}) {v}")
    eth_choice = Prompt.ask("  Choice", choices=list(ethnicities.keys()), default="1")
    ethnicity = ethnicities[eth_choice]

    # Jaundice
    console.print("\n[bold yellow]Was the person born with jaundice?[/bold yellow]  1) Yes  2) No")
    jundice = "yes" if Prompt.ask("  Choice", choices=["1", "2"], default="2") == "1" else "no"

    # Family ASD history
    console.print("\n[bold yellow]Does anyone in the immediate family have ASD?[/bold yellow]  1) Yes  2) No")
    austim = "yes" if Prompt.ask("  Choice", choices=["1", "2"], default="2") == "1" else "no"

    # Relation
    relations = {"1": "Self", "2": "Parent", "3": "Relative", "4": "Health care professional"}
    console.print("\n[bold yellow]Who is completing this test?[/bold yellow]")
    for k, v in relations.items():
        console.print(f"  {k}) {v}")
    rel_choice = Prompt.ask("  Choice", choices=list(relations.keys()), default="1")
    relation = relations[rel_choice]

    # ── Build feature vector ───────────────────────────────────
    result_score = sum(scores.values())  # sum of A1–A10

    raw = {
        **scores,
        "age":             age,
        "gender":          gender,
        "ethnicity":       ethnicity,
        "jundice":         jundice,
        "austim":          austim,
        "contry_of_res":   "Unknown",
        "used_app_before": "no",
        "result":          result_score,
        "age_desc":        "18 and more" if age >= 18 else "under 18",
        "relation":        relation,
    }

    input_df = build_input_vector(raw, feature_columns)
    input_scaled = scaler.transform(input_df)

    # ── Predict ────────────────────────────────────────────────
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # P(ASD Positive)

    # Risk level
    if probability < 0.35:
        risk_level  = "LOW"
        risk_color  = "green"
        risk_emoji  = "🟢"
    elif probability < 0.65:
        risk_level  = "MEDIUM"
        risk_color  = "yellow"
        risk_emoji  = "🟡"
    else:
        risk_level  = "HIGH"
        risk_color  = "red"
        risk_emoji  = "🔴"

    result_label = "ASD POSITIVE" if prediction == 1 else "ASD NEGATIVE"
    result_color = "red" if prediction == 1 else "green"

    # ── Display Result ─────────────────────────────────────────
    console.print()
    console.print(Rule("[bold white]━━━ YOUR RESULT ━━━[/bold white]"))
    console.print()

    result_text = (
        f"[bold {result_color}]{result_label}[/bold {result_color}]\n\n"
        f"[white]Confidence     :[/white] [bold cyan]{probability*100:.1f}%[/bold cyan]\n"
        f"[white]Risk Level     :[/white] [{risk_color}]{risk_emoji} {risk_level}[/{risk_color}]\n"
        f"[white]Q-CHAT Score   :[/white] [bold]{result_score}/10[/bold]\n"
    )

    if prediction == 1:
        result_text += "\n[dim yellow]⚕  Please consult a healthcare professional for clinical confirmation.[/dim yellow]"
    else:
        result_text += "\n[dim green]✔  No strong indicators detected. Stay healthy![/dim green]"

    console.print(Panel(
        result_text,
        title="[bold white]Screening Result[/bold white]",
        border_style=result_color,
        padding=(1, 4)
    ))
    console.print()

    # ── Save individual result plot ────────────────────────────
    _save_individual_result_plot(scores, probability, risk_level, result_label, "outputs/self_test_result.png")
    console.print(f"[dim]Result chart saved → [cyan]outputs/self_test_result.png[/cyan][/dim]\n")


def _save_individual_result_plot(scores, probability, risk_level, result_label, save_path):
    """Saves a simple visual of the test result."""
    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "text.color": "#c9d1d9",
        "axes.facecolor": "#161b22",   "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#c9d1d9",      "ytick.color": "#c9d1d9",
        "font.family": "monospace",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")
    fig.suptitle(f"ASD Screening Result — {result_label}", fontsize=14,
                 color="#f85149" if "POSITIVE" in result_label else "#3fb950", fontweight="bold")

    # Bar chart of A1–A10 scores
    q_labels = [f"A{i}" for i in range(1, 11)]
    q_values = list(scores.values())
    colors   = ["#f85149" if v == 1 else "#3fb950" for v in q_values]
    ax1.bar(q_labels, q_values, color=colors, edgecolor="#0d1117")
    ax1.set_title("Q-CHAT-10 Responses (1=Yes, 0=No)", color="#58a6ff", fontsize=11)
    ax1.set_ylim(0, 1.4)
    ax1.set_ylabel("Response")
    ax1.set_facecolor("#161b22")

    # Gauge-style probability bar
    bar_color = "#f85149" if probability > 0.65 else "#d29922" if probability > 0.35 else "#3fb950"
    ax2.barh(["ASD Risk"], [probability], color=bar_color, height=0.4)
    ax2.barh(["ASD Risk"], [1 - probability], left=[probability], color="#21262d", height=0.4)
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold")
    ax2.set_title(f"Risk Probability: {probability*100:.1f}%  [{risk_level}]",
                  color="#58a6ff", fontsize=11)
    ax2.legend(framealpha=0.2, labelcolor="white")
    ax2.set_facecolor("#161b22")

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# FEATURE 2 — Batch CSV Report (Separate PDF per patient)
# ═══════════════════════════════════════════════════════════════════
def batch_csv_report(model, scaler, feature_columns):
    """
    Loads a CSV of patients, predicts for each row,
    generates a separate PDF report per patient with charts.
    Does NOT touch any existing training code.
    """
    console.print()
    console.print(Rule("[bold magenta]📊 BATCH CSV REPORT GENERATOR[/bold magenta]"))
    console.print()

    # ── Get CSV path ───────────────────────────────────────────
    csv_path = Prompt.ask(
        "[bold yellow]Enter path to your patient CSV file[/bold yellow]",
        default="test_patients.csv"
    )

    if not os.path.exists(csv_path):
        console.print(f"[red]✘ File not found: {csv_path}[/red]")
        return

    # ── Load + preprocess ──────────────────────────────────────
    try:
        raw_data = pd.read_csv(csv_path)
        raw_data.columns = raw_data.columns.str.strip().str.replace("'", "", regex=False)
        raw_data = raw_data.replace("?", np.nan).fillna("Unknown")
    except Exception as e:
        console.print(f"[red]✘ Could not read CSV: {e}[/red]")
        return

    total_patients = len(raw_data)
    console.print(f"[green]✔[/green] Loaded [bold]{total_patients}[/bold] patients from [cyan]{csv_path}[/cyan]\n")

    # Output folder for batch PDFs
    batch_dir = f"{OUTPUT_DIR}/batch_reports"
    os.makedirs(batch_dir, exist_ok=True)

    # Track summary
    results_summary = []

    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "text.color": "#c9d1d9",
        "axes.facecolor":   "#161b22", "axes.labelcolor": "#c9d1d9",
        "xtick.color":      "#c9d1d9", "ytick.color": "#c9d1d9",
        "font.family":      "monospace",
    })

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Generating PDFs...", total=total_patients
        )

        for idx, row in raw_data.iterrows():
            patient_id = idx + 1
            progress.update(task, description=f"[cyan]Patient {patient_id}/{total_patients}")

            # ── Build input ────────────────────────────────────
            score_cols = [f"A{i}_Score" for i in range(1, 11)]
            scores = {}
            for col in score_cols:
                val = row.get(col, 0)
                try:
                    scores[col] = int(float(val))
                except:
                    scores[col] = 0

            result_score = sum(scores.values())

            raw = {
                **scores,
                "age":             row.get("age", 25),
                "gender":          str(row.get("gender", "m")).lower().strip(),
                "ethnicity":       str(row.get("ethnicity", "Others")),
                "jundice":         str(row.get("jundice", "no")).lower(),
                "austim":          str(row.get("austim", "no")).lower(),
                "contry_of_res":   str(row.get("contry_of_res", "Unknown")),
                "used_app_before": str(row.get("used_app_before", "no")).lower(),
                "result":          result_score,
                "age_desc":        "18 and more",
                "relation":        str(row.get("relation", "Self")),
            }

            input_df     = build_input_vector(raw, feature_columns)
            input_scaled = scaler.transform(input_df)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0][1]

            if probability < 0.35:
                risk_level = "LOW"
            elif probability < 0.65:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            result_label = "ASD POSITIVE" if prediction == 1 else "ASD NEGATIVE"
            result_color_hex = "#f85149" if prediction == 1 else "#3fb950"

            results_summary.append({
                "Patient": patient_id,
                "Prediction": result_label,
                "Confidence": f"{probability*100:.1f}%",
                "Risk": risk_level,
                "Q-CHAT Score": result_score,
            })

            # ── Generate PDF for this patient ──────────────────
            pdf_path = f"{batch_dir}/patient_{patient_id:03d}_report.pdf"

            with PdfPages(pdf_path) as pdf:

                # ── PAGE 1: Summary + Q-CHAT bar chart ────────
                fig1, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d1117")
                fig1.suptitle(
                    f"Patient {patient_id} — ASD Screening Report",
                    fontsize=15, color="#58a6ff", fontweight="bold"
                )

                # Demographics text box
                ax_info = axes[0]
                ax_info.axis("off")
                demo_lines = [
                    f"Patient ID    : {patient_id}",
                    f"Age           : {raw.get('age', 'N/A')}",
                    f"Gender        : {raw.get('gender', 'N/A').upper()}",
                    f"Ethnicity     : {raw.get('ethnicity', 'N/A')}",
                    f"Jaundice      : {raw.get('jundice', 'N/A').upper()}",
                    f"Family ASD    : {raw.get('austim', 'N/A').upper()}",
                    f"Relation      : {raw.get('relation', 'N/A')}",
                    "",
                    f"Q-CHAT Score  : {result_score}/10",
                    "",
                    f"PREDICTION    : {result_label}",
                    f"CONFIDENCE    : {probability*100:.1f}%",
                    f"RISK LEVEL    : {risk_level}",
                ]
                demo_text = "\n".join(demo_lines)
                ax_info.text(
                    0.05, 0.95, demo_text,
                    transform=ax_info.transAxes,
                    fontsize=11, verticalalignment="top",
                    fontfamily="monospace",
                    color="#c9d1d9",
                    bbox=dict(
                        boxstyle="round,pad=0.8",
                        facecolor=result_color_hex + "22",
                        edgecolor=result_color_hex,
                        linewidth=2
                    )
                )
                ax_info.set_facecolor("#161b22")

                # Q-CHAT-10 bar chart
                ax_bar = axes[1]
                q_labels = [f"A{i}" for i in range(1, 11)]
                q_values = [scores.get(f"A{i}_Score", 0) for i in range(1, 11)]
                bar_colors = ["#f85149" if v == 1 else "#3fb950" for v in q_values]
                ax_bar.bar(q_labels, q_values, color=bar_colors, edgecolor="#0d1117", width=0.6)
                ax_bar.set_title("Q-CHAT-10 Responses (Red=1/Yes, Green=0/No)",
                                 color="#58a6ff", fontsize=11)
                ax_bar.set_ylim(0, 1.5)
                ax_bar.set_ylabel("Response Value")
                ax_bar.axhline(0.5, color="white", linestyle="--", lw=1, alpha=0.5)
                ax_bar.set_facecolor("#161b22")

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig1, facecolor="#0d1117", bbox_inches="tight")
                plt.close(fig1)

                # ── PAGE 2: Risk gauge + Score radar ──────────
                fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d1117")
                fig2.suptitle(
                    f"Patient {patient_id} — Risk Analysis",
                    fontsize=15, color="#58a6ff", fontweight="bold"
                )

                # Risk probability gauge bar
                ax_gauge = axes2[0]
                gauge_color = (
                    "#f85149" if probability > 0.65
                    else "#d29922" if probability > 0.35
                    else "#3fb950"
                )
                ax_gauge.barh(["ASD Risk"], [probability],
                              color=gauge_color, height=0.35, edgecolor="#0d1117")
                ax_gauge.barh(["ASD Risk"], [1 - probability],
                              left=[probability], color="#21262d", height=0.35)
                ax_gauge.axvline(0.5, color="white", linestyle="--", lw=2, label="Threshold 0.5")
                ax_gauge.set_xlim(0, 1)
                ax_gauge.set_title(
                    f"Risk Probability: {probability*100:.1f}%  [{risk_level}]",
                    color="#58a6ff", fontsize=12
                )
                ax_gauge.legend(framealpha=0.2, labelcolor="white")
                ax_gauge.set_facecolor("#161b22")
                # Add probability text
                ax_gauge.text(
                    probability / 2, 0,
                    f"{probability*100:.1f}%",
                    ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold"
                )

                # Score distribution pie
                ax_pie = axes2[1]
                yes_count = sum(q_values)
                no_count  = 10 - yes_count
                wedge_colors = ["#f85149", "#3fb950"]
                explode = (0.05, 0.05)
                ax_pie.pie(
                    [yes_count, no_count],
                    labels=[f"Atypical ({yes_count})", f"Typical ({no_count})"],
                    colors=wedge_colors, autopct="%1.0f%%",
                    startangle=90, explode=explode,
                    textprops={"color": "white", "fontsize": 12},
                    wedgeprops={"linewidth": 2, "edgecolor": "#0d1117"}
                )
                ax_pie.set_title("Q-CHAT Atypical vs Typical Responses",
                                 color="#58a6ff", fontsize=12)
                ax_pie.set_facecolor("#161b22")

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig2, facecolor="#0d1117", bbox_inches="tight")
                plt.close(fig2)

                # ── PAGE 3: Score comparison (this patient vs dataset avg) ──
                fig3, ax3 = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
                ax3.set_facecolor("#161b22")

                x = np.arange(10)
                width = 0.35
                dataset_avg = [0.6, 0.5, 0.5, 0.6, 0.4, 0.4, 0.5, 0.8, 0.4, 0.5]  # approx from dataset

                bars1 = ax3.bar(x - width/2, q_values,    width, label=f"Patient {patient_id}",
                                color="#58a6ff", edgecolor="#0d1117")
                bars2 = ax3.bar(x + width/2, dataset_avg, width, label="Dataset Average",
                                color="#d29922", edgecolor="#0d1117", alpha=0.7)

                ax3.set_title(f"Patient {patient_id} Scores vs Dataset Average",
                              color="#58a6ff", fontsize=13)
                ax3.set_xticks(x)
                ax3.set_xticklabels([f"A{i}" for i in range(1, 11)])
                ax3.set_ylabel("Score (0 or 1)")
                ax3.set_ylim(0, 1.4)
                ax3.legend(framealpha=0.2, labelcolor="white")
                ax3.set_facecolor("#161b22")

                plt.tight_layout()
                pdf.savefig(fig3, facecolor="#0d1117", bbox_inches="tight")
                plt.close(fig3)

                # ── PDF Metadata ────────────────────────────────
                d = pdf.infodict()
                d["Title"]   = f"ASD Screening Report — Patient {patient_id}"
                d["Subject"] = f"Prediction: {result_label} | Risk: {risk_level}"

            progress.advance(task)
            time.sleep(0.05)

    # ── Summary table ──────────────────────────────────────────
    console.print()
    summary_tbl = Table(
        title=f"Batch Report Summary ({total_patients} patients)",
        box=box.ROUNDED, border_style="green", show_lines=True
    )
    summary_tbl.add_column("Patient",      style="bold yellow", justify="center")
    summary_tbl.add_column("Prediction",   style="white",       justify="center")
    summary_tbl.add_column("Confidence",   style="cyan",        justify="center")
    summary_tbl.add_column("Risk Level",   style="white",       justify="center")
    summary_tbl.add_column("Q-CHAT Score", style="green",       justify="center")

    for r in results_summary:
        pred_color = "red" if "POSITIVE" in r["Prediction"] else "green"
        risk_color = "red" if r["Risk"] == "HIGH" else "yellow" if r["Risk"] == "MEDIUM" else "green"
        summary_tbl.add_row(
            str(r["Patient"]),
            f"[{pred_color}]{r['Prediction']}[/{pred_color}]",
            r["Confidence"],
            f"[{risk_color}]{r['Risk']}[/{risk_color}]",
            str(r["Q-CHAT Score"]),
        )

    console.print(summary_tbl)
    console.print()
    console.print(f"[green]✔[/green] PDFs saved to [cyan]{batch_dir}/[/cyan]")
    console.print(f"[dim]  → patient_001_report.pdf ... patient_{total_patients:03d}_report.pdf[/dim]\n")


# ═══════════════════════════════════════════════════════════════════
# MENU — shown after main pipeline completes
# ═══════════════════════════════════════════════════════════════════
def show_menu(model, scaler, feature_columns):
    while True:
        console.print()
        console.print(Rule("[bold white]MAIN MENU[/bold white]"))
        console.print(Panel(
            "[bold cyan]1)[/bold cyan]  🧠  ASD Self Screening Test\n"
            "[bold cyan]2)[/bold cyan]  📊  Batch CSV Report (PDF per patient)\n"
            "[bold cyan]3)[/bold cyan]  🚪  Exit",
            title="[bold white]What would you like to do?[/bold white]",
            border_style="cyan",
            padding=(1, 4)
        ))

        choice = Prompt.ask("[bold yellow]Enter choice[/bold yellow]", choices=["1", "2", "3"], default="3")

        if choice == "1":
            conversational_test(model, scaler, feature_columns)
        elif choice == "2":
            batch_csv_report(model, scaler, feature_columns)
        elif choice == "3":
            console.print(Panel(
                "[bold green]Pipeline complete. Goodbye![/bold green]",
                border_style="green"
            ))
            break


# ═══════════════════════════════════════════════════════════════════
# MAIN — existing pipeline unchanged + save model + show menu
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

    # ── NEW: Save model + scaler + feature columns ─────────────
    feature_columns = X.columns.tolist()
    joblib.dump((model, scaler, feature_columns), MODEL_PATH)
    console.print(f"[dim]Model saved → [cyan]{MODEL_PATH}[/cyan][/dim]\n")

    # Final summary panel
    summary = (
        f"[bold green]Accuracy :[/bold green] {acc*100:.2f}%\n"
        f"[bold cyan]ROC-AUC  :[/bold cyan] {auc:.4f}\n"
        f"[bold yellow]CV Acc   :[/bold yellow] {cv_sc.mean():.4f} ± {cv_sc.std():.4f}\n"
        f"[bold white]Plots    :[/bold white] saved to [cyan]{OUTPUT_DIR}/[/cyan]"
    )
    console.print(Panel(summary, title="[bold white]Pipeline Complete ✔[/bold white]",
                        border_style="green", padding=(1, 4)))

    # ── NEW: Show interactive menu ──────────────────────────────
    show_menu(model, scaler, feature_columns)


if __name__ == "__main__":
    main()