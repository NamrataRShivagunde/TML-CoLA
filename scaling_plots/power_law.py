import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




def fit_power_law(model_sizes, performance):
    """
    Fit y = a * x^b using scipy.optimize.curve_fit on original (x, y),
    initialized from a log-log linear regression. Returns (a, b).
    """
    x = np.array(model_sizes, dtype=float)
    y = np.array(performance, dtype=float)

    def power_fn(xx, aa, bb, cc):
        return aa * np.power(xx, bb) + cc

    popt, _ = curve_fit(power_fn, x, y, maxfev=1000000) # maxfev is max calls to function
    a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
    return a, b, c


def predict_performance(model_size, a, b, c):
    """Predict performance for scalar or array model_size.
    Supports numpy arrays by using vectorized operations.
    """
    x = np.asarray(model_size, dtype=float)
    return float(a) * np.power(x, float(b)) + float(c)


def plot_power_law_curvefit(model_sizes, performance, title, outfile):
    """
    Scatter observations and overlay fitted curve using direct y = a * x^b fit.
    """
    a, b, c = fit_power_law(model_sizes, performance)

    # Smooth x-range for the fitted curve in linear space (include prediction points)
    pred_sizes = np.array([1e9, 7e9])  # 1B and 7B
    x_min = min(np.min(model_sizes), np.min(pred_sizes))
    x_max = max(np.max(model_sizes), np.max(pred_sizes))
    xs = np.linspace(x_min, x_max, 400)
    ys = predict_performance(xs, a, b, c)

    plt.figure(figsize=(7, 5))
    plt.plot(model_sizes, performance, 'o', label='observations')
    plt.plot(xs, ys, '-', label=f'curve_fit: y = {a:.4f} * x^{b:.4f} + {c:.4f}')

    # Predicted markers at 1B and 7B with stars and dashed guide lines
    pred_vals = predict_performance(pred_sizes, a, b, c)
    labels = ['pred 1B', 'pred 7B']
    for px, py, lab in zip(pred_sizes, pred_vals, labels):
        plt.plot(px, py, marker='*', markersize=12, linestyle='None', label=lab)
        plt.axvline(px, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.axhline(py, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.annotate(f"{lab}: {py:.2f} (= ppl {np.exp(py):.2f})", xy=(px, py), xytext=(6, 6), textcoords='offset points')
    plt.xlabel('Model size (parameters)')
    plt.ylabel('Eval loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Saved plot: {outfile} | curve_fit params a={a:.6f}, b={b:.6f}, c={c:.6f}")


def plot_power_law_curvefit_log(model_sizes, performance, title, outfile):
    """
    Log-log plot: scatter observations and overlay fitted curve using y = a * x^b.
    Includes star markers and dashed guides for 1B and 7B predictions.
    """
    a, b, c = fit_power_law(model_sizes, performance)

    # Smooth x-range in log-space; extend to include prediction points
    pred_sizes = np.array([1e9, 7e9])
    x_min = min(np.min(model_sizes), np.min(pred_sizes))
    x_max = max(np.max(model_sizes), np.max(pred_sizes))
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 400)
    ys = predict_performance(xs, a, b, c)

    plt.figure(figsize=(7, 5))
    plt.loglog(model_sizes, performance, 'o', label='observations')
    plt.loglog(xs, ys, '-', label=f'log-fit: y = {a:.4f} * x^{b:.4f} + {c:.4f}')

    # Predicted markers at 1B and 7B
    pred_vals = predict_performance(pred_sizes, a, b, c)
    labels = ['pred 1B', 'pred 7B']
    for px, py, lab in zip(pred_sizes, pred_vals, labels):
        plt.loglog(px, py, marker='*', markersize=12, linestyle='None', label=lab)
        plt.axvline(px, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.axhline(py, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.annotate(f"{lab}: {py:.2f}(= ppl {np.exp(py):.2f})", xy=(px, py), xytext=(6, 6), textcoords='offset points')

    plt.xlabel('Model size (parameters)')
    plt.ylabel('Eval loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Saved log plot: {outfile} | log-fit params a={a:.6f}, b={b:.6f}, c={c:.6f}")


if __name__ == "__main__":
    # Provided measurements (perplexity) by model size
    # Sizes: 60M, 130M, 350M
    sizes = [60e6, 130e6, 350e6]

    # Baseline perplexities
    baseline_ppl = [
        29.550375401273623,  # 60M baseline
        21.87082859529418,   # 130M baseline
        19.16091547584335,   # 350M baseline
    ]

    # CoLA perplexities
    cola_ppl = [
        30.255232987651407,  # 60M CoLA
        21.509907901762894,  # 130M CoLA (3k warmup)
        18.245290801722234,  # 350M CoLA (10k warmup)
    ]

    baseline_eval_loss = [np.log(x) for x in baseline_ppl]
    cola_eval_loss = [np.log(x) for x in cola_ppl]
    print("Baseline eval losses:", baseline_eval_loss)
    print("CoLA eval losses:", cola_eval_loss)

    # Fit power-law (log-fit) for reference
    a, b, c = fit_power_law(sizes, baseline_eval_loss)
    a_cola, b_cola, c_cola = fit_power_law(sizes, cola_eval_loss)

    print("\nFitted parameters (Baseline):", (a, b, c))
    print("Fitted parameters (CoLA):", (a_cola, b_cola, c_cola))

    # # Predictions at 1B and 7B only
    target_sizes = np.array([1e9, 7e9])  # 1B, 7B
    preds_baseline_loss = predict_performance(target_sizes, a, b, c)
    preds_cola_loss = predict_performance(target_sizes, a_cola, b_cola, c_cola)

    # print eval loss for 1b and 7B
    print("\nBaseline eval loss at 1B: {:.4f}, at 7B: {:.4f}".format(preds_baseline_loss[0], preds_baseline_loss[1]))
    print("CoLA eval loss at 1B: {:.4f}, at 7B: {:.4f}".format(preds_cola_loss[0], preds_cola_loss[1]))
    print("-----------------------------------------------------")

    print("\nBaseline eval perplexity at 1B: {:.4f}, at 7B: {:.4f}".format(np.exp(preds_baseline_loss[0]), np.exp(preds_baseline_loss[1])))
    print("CoLA eval perplexity at 1B: {:.4f}, at 7B: {:.4f}".format(np.exp(preds_cola_loss[0]), np.exp(preds_cola_loss[1])))
    print("-----------------------------------------------------")

    # plot
    # # # Plots with curve_fit (direct y = a * x^b)
    plot_power_law_curvefit(sizes, baseline_eval_loss, title='Baseline Eval loss', outfile='baseline_curvefit.png')
    plot_power_law_curvefit(sizes, cola_eval_loss, title='CoLA Eval loss', outfile='cola_curvefit.png')

    # log-log plots using the log-space fit
    plot_power_law_curvefit_log(sizes, baseline_eval_loss, title='Baseline (log-log)', outfile='baseline_curvefit_log.png')
    plot_power_law_curvefit_log(sizes, cola_eval_loss, title='CoLA (log-log)', outfile='cola_curvefit_log.png')