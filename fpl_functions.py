# filepath: /Users/nasserm/Documents/vscode/lsstda/bayesian-fpl/fpl_functions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.stats import norm
import corner # Added import

class player():
    def __init__(self, gw_data_path, teams_data_path='teams.csv', player_opponent_team_id_col='opponent_team', player_total_points_col='total_points', team_id_col='id', team_strength_col='strength_defence_home'):
        '''
        Initializes the player object with data and prepares it for MCMC analysis.

        Parameters:
        -----------
        gw_data_path : str
            Path to the CSV file containing gameweek data (e.g., 'gw.csv').
            This file should have columns for opponent team ID and total points.
        teams_data_path : str
            Path to the CSV file containing team data (e.g., 'teams.csv').
            This file should have columns for team ID and their defensive strength.
        player_opponent_team_id_col : str, optional
            Column name in gw_data_path for the opponent team ID.
        player_total_points_col : str, optional
            Column name in gw_data_path for the total points scored by the player.
        team_id_col : str, optional
            Column name in teams_data_path for the team ID.
        team_strength_col : str, optional
            Column name in teams_data_path for the team's defensive strength.
        '''
        df = pd.read_csv(gw_data_path)
        self.data = df[[player_opponent_team_id_col, player_total_points_col]]
        self.data.rename(columns={player_opponent_team_id_col: 'opponent_team', player_total_points_col: 'total_points'}, inplace=True)

        teams_df = pd.read_csv(teams_data_path)
        
        # Map opponent team strength to fixture
        self.data['fixture'] = self.data['opponent_team'].map(teams_df.set_index(team_id_col)[team_strength_col])

        # Calculate form (average points of last 4 games)
        form = []
        for i in range(0, len(self.data)):
            if i < 4: # Use available history if less than 4 games
                form.append(np.mean(self.data['total_points'].iloc[:i+1]) if i > 0 else 0)
            else:
                form.append(np.mean(self.data['total_points'].iloc[i-4:i]))
        self.data['form'] = form
        
        self.total_points = self.data['total_points'].values
        self.form = self.data['form'].values
        self.fixture = self.data['fixture'].values
        self.y_unc = self.data['total_points'].std()

        self.ndim = 4  # alpha, beta_1, beta_2, sigma
        self.nwalkers = 100
        self.sampler = None
        self.flat_samples = None
        self.labels = ["alpha", "beta_form", "beta_fixture", "sigma"]


    def get_model_predictions(self, theta, form, fixture): # Made it a method
        alpha, beta_1, beta_2, sigma = theta
        mu = alpha + beta_1*form + beta_2*fixture
        sigma = np.abs(sigma) # Ensure sigma is positive
        draw = norm.rvs(loc=mu, scale=sigma)
        return draw

    def _lnprior(self, theta):
        alpha, beta_1, beta_2, sigma = theta
        # Priors: alpha, beta_1, beta_2 ~ U(-100, 100), sigma ~ U(0, 100)
        if -100 <= alpha <= 100 and -100 <= beta_1 <= 100 and -100 <= beta_2 <= 100 and 0 <= sigma <= 100:
            # Log of uniform prior: log(1/range)
            return np.log(1/200) + np.log(1/200) + np.log(1/200) + np.log(1/100)
        return -np.inf

    def _lnlikelihood(self, theta):
        model_preds = self.get_model_predictions(theta, self.form, self.fixture)
        # Assuming Gaussian likelihood
        lnl = -0.5 * np.sum(((self.total_points - model_preds) / self.y_unc)**2 + np.log(2 * np.pi * self.y_unc**2))
        return lnl

    def _lnprob(self, theta):
        lp = self._lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._lnlikelihood(theta)

    def run_mcmc(self, n_steps=5000, n_burnin=1000, progress=True):
        '''
        Runs the MCMC sampler.

        Parameters:
        -----------
        n_steps : int
            Number of MCMC steps to run.
        n_burnin : int
            Number of steps to discard as burn-in.
        progress : bool
            Whether to show a progress bar.
        '''
        # Initial guess for parameters (can be randomized or informed)
        alpha0 = np.random.uniform(-10, 10, self.nwalkers)
        beta_1_0 = np.random.uniform(-5, 5, self.nwalkers)
        beta_2_0 = np.random.uniform(-5, 5, self.nwalkers)
        sigma0 = np.random.uniform(0.1, self.data['total_points'].std() * 2, self.nwalkers)
        p0 = np.array([alpha0, beta_1_0, beta_2_0, sigma0]).T
        
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self._lnprob)
        self.sampler.run_mcmc(p0, n_steps, progress=progress)
        
        self.flat_samples = self.sampler.get_chain(discard=n_burnin, thin=15, flat=True)
        
        print("MCMC run complete.")
        self._calculate_r_hat() # Calculate and print R-hat

    def _calculate_r_hat(self):
        if self.sampler is None:
            print("MCMC sampler has not been run yet.")
            return

        print("Gelman-Rubin diagnostic (R-hat values):")
        try:
            chain = self.sampler.get_chain(discard=100) # Use a portion for R-hat
            for i, label in enumerate(self.labels):
                # Reshape to (n_steps, n_walkers)
                param_chain = chain[:, :, i]
                
                # Calculate within-chain variance (W)
                W = np.mean(np.var(param_chain, axis=0, ddof=1))
                
                # Calculate between-chain variance (B)
                chain_means = np.mean(param_chain, axis=0)
                B_over_n = np.var(chain_means, ddof=1)
                B = B_over_n * param_chain.shape[0] # n_samples per chain
                
                if W == 0: # Avoid division by zero if W is zero
                    r_hat = np.nan # Or handle as appropriate
                else:
                    # Estimate of marginal posterior variance
                    var_hat = ((param_chain.shape[0] - 1) / param_chain.shape[0]) * W + (B / (param_chain.shape[0] * self.nwalkers)) # Corrected B term
                    r_hat = np.sqrt(var_hat / W)
                
                print(f"{label}: {r_hat:.4f}")
        except Exception as e:
            print(f"Could not calculate R-hat: {e}")


    def plot_chains(self):
        '''Plots the MCMC chains for each parameter.'''
        if self.sampler is None:
            print("MCMC sampler has not been run yet. Run run_mcmc() first.")
            return

        samples = self.sampler.get_chain()
        
        fig, axes = plt.subplots(self.ndim, figsize=(10, 2 * self.ndim), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
        axes[-1].set_xlabel("Step Number")
        plt.suptitle("MCMC Chains")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        plt.show()

    def plot_corner(self):
        '''Generates a corner plot of the posterior distributions.'''
        if self.flat_samples is None:
            print("Flat samples not available. Run run_mcmc() first.")
            return
        
        fig = corner.corner(self.flat_samples, labels=self.labels, truths=[np.median(self.flat_samples[:,i]) for i in range(self.ndim)])
        plt.suptitle("Corner Plot of Posterior Distributions")
        plt.show()

    def predict_points(self, next_fixture_strength, n_draws=1000):
        '''
        Predicts points for a future game using the MCMC results.

        Parameters:
        -----------
        next_fixture_strength : float or int
            The defensive strength of the opponent in the next game.
        n_draws : int
            Number of samples to draw for the prediction.

        Returns:
        --------
        tuple
            (mean_predicted_points, std_predicted_points, predicted_samples)
        '''
        if self.flat_samples is None:
            print("Flat samples not available. Run run_mcmc() first.")
            return None, None, None

        # Get median parameter values from the posterior
        alpha_med = np.median(self.flat_samples[:, 0])
        beta1_med = np.median(self.flat_samples[:, 1])
        beta2_med = np.median(self.flat_samples[:, 2])
        sigma_med = np.median(self.flat_samples[:, 3])
        
        theta_median = [alpha_med, beta1_med, beta2_med, sigma_med]
        
        # Use the most recent form value
        last_form = self.form[-1] if len(self.form) > 0 else 0
        
        predicted_samples = np.array([self.get_model_predictions(theta_median, last_form, next_fixture_strength) for _ in range(n_draws)])
        
        # We can also sample from the posterior for theta for a more robust prediction
        # For each sample theta from flat_samples, draw a point prediction
        # This incorporates parameter uncertainty
        
        # For simplicity, using median parameters first as in the notebook.
        # To incorporate parameter uncertainty:
        # predicted_samples_full_posterior = []
        # for i in range(min(n_draws, len(self.flat_samples))): # Draw up to n_draws or available samples
        #     theta_sample = self.flat_samples[np.random.choice(len(self.flat_samples))] 
        #     pred = self.get_model_predictions(theta_sample, last_form, next_fixture_strength)
        #     predicted_samples_full_posterior.append(pred)
        # predicted_samples = np.array(predicted_samples_full_posterior)


        mean_predicted = np.mean(predicted_samples)
        std_predicted = np.std(predicted_samples)
        
        print(f"Predicted points for next game (fixture strength {next_fixture_strength}):")
        print(f"Mean: {mean_predicted:.2f}, Std: {std_predicted:.2f}")
        
        return mean_predicted, std_predicted, predicted_samples

    def plot_prediction_histogram(self, predicted_samples, player_name="Player"):
        '''
        Plots a histogram of the predicted points.

        Parameters:
        -----------
        predicted_samples : array-like
            Array of predicted point samples.
        player_name : str
            Name of the player for the plot title.
        '''
        if predicted_samples is None or len(predicted_samples) == 0:
            print("No predicted samples to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(predicted_samples, bins=30, density=True, alpha=0.7, label=f'Predicted Points (Next Game)')
        
        # Optionally, overlay historical points
        if len(self.total_points) > 0:
            ax.hist(self.total_points, bins=np.arange(0, np.max(self.total_points) + 2) - 0.5, density=True, alpha=0.5, label='Historical Points')
        
        ax.set_xlabel('Total Points')
        ax.set_ylabel('Density')
        ax.set_title(f'Predicted Points Distribution for {player_name}')
        ax.legend()
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()