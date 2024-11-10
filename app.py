from flask import Flask, render_template, request, session, url_for
import numpy as np
import matplotlib
from scipy.stats import t
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

# Data generation function with a structured approach
def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.rand(N)
    Y = beta0 + beta1 * X + mu + np.random.normal(0, sigma2, N)

    # Fit a linear regression model
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Scatter plot with regression line
    plt.scatter(X, Y, color="blue", label="Data")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.clf()

    # Simulate multiple slopes and intercepts
    slopes, intercepts = [], []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, sigma2, N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Histogram of slopes and intercepts
    plt.hist(slopes, bins=20, alpha=0.7, color="blue", label="Slopes")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.clf()

    # Proportion of more extreme slopes and intercepts
    slope_more_extreme = np.mean(np.abs(slopes) > np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) > np.abs(intercept))

    return (
        X, Y, slope, intercept, plot1_path, plot2_path,
        slope_more_extreme, intercept_extreme, slopes, intercepts
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve form inputs
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and plots
        (
            X, Y, slope, intercept, plot1, plot2,
            slope_extreme, intercept_extreme, slopes, intercepts
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session for later use
        session.update({
            "X": X.tolist(), "Y": Y.tolist(),
            "slope": slope, "intercept": intercept,
            "slopes": slopes, "intercepts": intercepts,
            "slope_extreme": slope_extreme, "intercept_extreme": intercept_extreme,
            "N": N, "mu": mu, "sigma2": sigma2, "beta0": beta0, "beta1": beta1, "S": S
        })

        return render_template(
            "index.html", plot1=plot1, plot2=plot2,
            slope_extreme=slope_extreme, intercept_extreme=intercept_extreme,
            N=N, mu=mu, sigma2=sigma2, beta0=beta0, beta1=beta1, S=S
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    session.clear()
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = session.get("N")
    slope, intercept = session.get("slope"), session.get("intercept")
    slopes, intercepts = session.get("slopes"), session.get("intercepts")
    beta0, beta1 = session.get("beta0"), session.get("beta1")

    # Get test parameters from form
    parameter, test_type = request.form.get("parameter"), request.form.get("test_type")

    # Select appropriate values based on parameter
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat, hypothesized_value = slope, beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat, hypothesized_value = intercept, beta0

    # Calculate p-value based on test type
    if test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats) >= np.abs(observed_stat))
    elif test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    else:
        p_value = np.mean(simulated_stats <= observed_stat)

    fun_message = "Extremely significant result!" if p_value <= 0.0001 else None

    # Plot hypothesis testing results
    plt.hist(simulated_stats, bins=20, color="lightgray")
    plt.axvline(observed_stat, color="red", linestyle="--", label="Observed Stat")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.clf()

    return render_template(
        "index.html",
        plot1="static/plot1.png", plot2="static/plot2.png", plot3=plot3_path,
        parameter=parameter, observed_stat=observed_stat,
        hypothesized_value=hypothesized_value, N=N,
        beta0=beta0, beta1=beta1, p_value=p_value, fun_message=fun_message
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        # Retrieve session data
        N, beta0, beta1 = session.get("N"), session.get("beta0"), session.get("beta1")
        parameter, confidence_level = request.form.get("parameter"), float(request.form.get("confidence_level"))

        # Select estimates based on chosen parameter
        if parameter == "slope":
            estimates = np.array(session.get("slopes"))
            true_param = beta1
        else:
            estimates = np.array(session.get("intercepts"))
            true_param = beta0

        # Calculate mean and standard error
        mean_estimate = np.mean(estimates)
        std_error = np.std(estimates, ddof=1) / np.sqrt(N)

        # Determine t critical value for confidence level
        alpha, df = 1 - confidence_level / 100, N - 1
        t_value = t.ppf(1 - alpha / 2, df)

        # Calculate confidence interval bounds
        ci_lower = mean_estimate - t_value * std_error
        ci_upper = mean_estimate + t_value * std_error
        includes_true = ci_lower <= true_param <= ci_upper

        # Plot confidence interval
        plt.figure(figsize=(8, 6))
        plt.scatter(estimates, [0] * len(estimates), color="gray", alpha=0.6, label="Simulated Estimates")
        plt.axvline(true_param, color="green", linestyle="--", linewidth=2.5, label="True " + parameter.capitalize())
        plt.plot([ci_lower, ci_upper], [0, 0], color="blue", linewidth=5, label=f"{confidence_level}% Confidence Interval")
        plt.plot(mean_estimate, 0, 'o', color="blue", markersize=10, label="Mean Estimate")
        plt.xlabel(f"{parameter.capitalize()} Estimate")
        plt.yticks([])
        plt.title(f"{confidence_level}% Confidence Interval for {parameter.capitalize()} (Mean Estimate)")
        plt.legend(loc="upper right", frameon=True, framealpha=1, edgecolor="black")

        # Save the plot
        plot4_path = "static/plot4.png"
        plt.savefig(plot4_path, bbox_inches='tight')
        plt.clf()

        # Render template with results
        return render_template(
            "index.html",
            plot1="static/plot1.png", plot2="static/plot2.png", plot4=plot4_path,
            parameter=parameter, confidence_level=confidence_level,
            mean_estimate=mean_estimate, ci_lower=ci_lower, ci_upper=ci_upper,
            includes_true=includes_true
        )
    except Exception as e:
        print("An error occurred:", e)
        return "An error occurred while calculating the confidence interval."

if __name__ == "__main__":
    app.run(debug=True)
