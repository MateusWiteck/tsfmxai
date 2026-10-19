"""Deterministic scientific diagrams of the four current generator strategies."""

ALGORITHMS = {
    "original": (
        "Original random AST",
        [("Derive local seed from root seed and mechanism index", False),
         ("Sample the number of binary operators", False),
         ("Sample parameter and lag leaves (time optional)", False),
         ("Sample Catalan-weighted binary tree shape", False),
         ("Label ADD/MULTIPLY operators", False),
         ("Canonicalize and bind coefficients and lags", False),
         ("Retain the executable mechanism", False),
         ("Sample a uniform initial history", False),
         ("Execute AST and add Gaussian innovations", False),
         ("Record trajectory and first numerical failure", False)],
    ),
    "jacobian": (
        "Jacobian contraction on an invariant box",
        [("Derive local seed from root seed and mechanism index", False),
         ("Sample original tree topology and bindings", False),
         ("Apply configured lag coefficient gates", True),
         ("Bound values and derivatives on the state box", True),
         ("Include lag-shift rows in weighted Jacobian norm", True),
         ("Scale AST for contraction and domain invariance", True),
         ("Require norm bound < 1 and noisy output inside box", True),
         ("Initialize history inside the certified box", True),
         ("Add bounded uniform innovations", True),
         ("Execute AST and check domain invariance", True)],
    ),
    "shao-wu": (
        "Shao-Wu: globally bounded nonlinear ASTs",
        [("Derive local seed from root seed and mechanism index", False),
         ("Sample binary trees, coefficients and lags", False),
         ("Wrap each lag in TANH; one equation per variable", True),
         ("Propagate global value and derivative bounds", True),
         ("Sum Euclidean lag-block Lipschitz bounds", True),
         ("Scale all equations to total gain < 1", True),
         ("Certify global lag gain and finite second moment", True),
         ("Draw a uniform initial lag history directly", True),
         ("Execute AST with independent future innovations", False),
         ("Retain final formulas, certificate and trajectories", True)],
    ),
    "heaps-var": (
        "Heaps: symmetric stationary VAR(p)",
        [("Derive local seed from root seed and mechanism index", False),
         ("Sample order p and free matrices U_l", True),
         ("Map U_l to admissible partial autocorrelations", True),
         ("Backward recursion: stationary marginal covariance", True),
         ("Forward/backward recursion: VAR coefficients", True),
         ("Assemble joint lag-state covariance", True),
         ("Check companion radius and Lyapunov residual", True),
         ("Store coefficients as one linear AST per variable", True),
         ("Draw the joint stationary Gaussian history", True),
         ("Execute AST and add Gaussian innovations", False)],
    ),
}


def plot_generator_algorithm(method: str):
    """Draw ten steps with red additions/replacements relative to original AST."""
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    title, steps = ALGORITHMS[method]
    fig, ax = plt.subplots(figsize=(16, 7), dpi=125)
    ax.set(xlim=(0, 16), ylim=(0, 7))
    ax.axis("off")
    ax.text(.35, 6.6, title, fontsize=20, fontweight="bold", va="top")
    ax.text(.35, 6.0, "Implemented sampling and execution; red marks changes relative to the original workflow.",
            fontsize=11, color="#667085")
    width, height = 2.55, 1.45
    positions = [(1.7 + 3.1 * i, 4.6) for i in range(5)]
    positions += [(14.1 - 3.1 * i, 1.9) for i in range(5)]
    for i in range(9):
        x, y = positions[i]
        nx, ny = positions[i + 1]
        if y == ny:
            sign = 1 if nx > x else -1
            start, end = (x + sign * width / 2, y), (nx - sign * width / 2, ny)
        else:
            start, end = (x, y - height / 2), (nx, ny + height / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                     linewidth=2, color="#c62828" if steps[i+1][1] else "#667085"))
    for i, ((label, changed), (x, y)) in enumerate(zip(steps, positions)):
        color = "#c62828" if changed else "#232323"
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
            boxstyle="round,pad=.07", facecolor="#ffebee" if changed else "#f4f6f8",
            edgecolor=color, linewidth=2))
        ax.text(x, y, textwrap.fill(f"{i+1}. {label}", width=26), ha="center", va="center",
                fontsize=10.2, color=color, fontweight="bold" if changed else "normal")
    ax.legend(handles=[Patch(facecolor="#f4f6f8", edgecolor="#232323", label="Shared operation"),
                       Patch(facecolor="#ffebee", edgecolor="#c62828", label="Added or replaced operation")],
              loc="lower center", ncol=2, frameon=False)
    fig.tight_layout()
    return fig
