"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data

#modify SDT MODEL
def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    #C = len(data['condition'].unique())     ### old code
    # stimulus_type and difficulty can be extracted from condition index:
    # condition: 0 to 3, mapping is difficulty*2 + stimulus_type
    # So, stimulus_type = condition % 2, difficulty = condition // 2
    stimulus_type = data['condition'] % 2
    difficulty = data['condition'] // 2
    
    # Map each data row to its stimulus and difficulty
    stimulus_type = stimulus_type.values
    difficulty = difficulty.values
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level intercepts
        mu_d_prime = pm.Normal('mu_d_prime', mu=0, sigma=1)
        mu_criterion = pm.Normal('mu_criterion', mu=0, sigma=1)
        
        # Group-level effects
        beta_stimulus_d_prime = pm.Normal('beta_stimulus_d_prime', 0, 1)
        beta_difficulty_d_prime = pm.Normal('beta_difficulty_d_prime', 0, 1)
        
        beta_stimulus_criterion = pm.Normal('beta_stimulus_criterion', 0, 1)
        beta_difficulty_criterion = pm.Normal('beta_difficulty_criterion', 0, 1)
        
        # Individual variability (participant-level deviations)
        sigma_d_prime = pm.HalfNormal('sigma_d_prime', 1)
        sigma_criterion = pm.HalfNormal('sigma_criterion', 1)
        
        # Participant-level random effects
        d_prime_participant = pm.Normal('d_prime_participant', 0, sigma_d_prime, shape=P)
        criterion_participant = pm.Normal('criterion_participant', 0, sigma_criterion, shape=P)
        
        # Expected d' and criterion per observation (row in data)
        d_prime = (mu_d_prime +
                   beta_stimulus_d_prime * stimulus_type +
                   beta_difficulty_d_prime * difficulty +
                   d_prime_participant[data['pnum'] - 1])
        
        criterion = (mu_criterion +
                     beta_stimulus_criterion * stimulus_type +
                     beta_difficulty_criterion * difficulty +
                     criterion_participant[data['pnum'] - 1])
        
        # Compute hit and false alarm rates with inverse logit
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
        
        # Likelihood for hits (signal trials)
        pm.Binomial('hit_obs',
                    n=data['nSignal'],
                    p=hit_rate,
                    observed=data['hits'])
        
        # Likelihood for false alarms (noise trials)
        pm.Binomial('false_alarm_obs',
                    n=data['nNoise'],
                    p=false_alarm_rate,
                    observed=data['false_alarms'])

    return sdt_model

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            label1 = CONDITION_NAMES.get(conditions[i], f'Condition {conditions[i]}')
            label2 = CONDITION_NAMES.get(conditions[j], f'Condition {conditions[j]}')

            axes[i,j].text(50, -0.27, 
                        f'{label1} - {label2}', 
                        ha='center', va='top', fontsize=12)
            axes[j,i].text(50, -0.27, 
                        f'{label2} - {label1}', 
                        ha='center', va='top', fontsize=12)

            plt.tight_layout()
            
    # Save the figure, print figure out!!!
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')
    plt.show()

# Main execution
#if __name__ == "__main__":
    #file_to_print = Path(__file__).parent / 'README.md'
    #with open(file_to_print, 'r') as file:
        #print(file.read())

#   MODIFIED main execution  -> I acknowledge the use of AI/ChatGPT
if __name__ == "__main__":
    
    # Step 1: Load and prepare SDT data
    sdt_data = read_data('data.csv', prepare_for='sdt', display=True)
    
    # Step 2: Build and sample the SDT model
    model = apply_hierarchical_sdt_model(sdt_data)
    with model:
        idata = pm.sample(draws=300, tune=300, target_accept=0.9, 
                          chains=2, cores=2, return_inferencedata=True)
    
    # Step 3: Check convergence (Part 2a)
    summary = az.summary(idata, round_to=2)
    print("===== Posterior Summary =====")
    print(summary)

    # Optionally check R-hat and effective sample size specifically:
    convergence_check = az.rhat(idata)
    print("===== R-hat (Convergence Diagnostic) =====")
    print(convergence_check)

    # Step 4: Posterior distribution plots (Part 2b)
    az.plot_posterior(idata, 
    var_names=[
        "mu_d_prime", "mu_criterion",
        "beta_stimulus_d_prime", "beta_difficulty_d_prime",
        "beta_stimulus_criterion", "beta_difficulty_criterion"
    ],
    hdi_prob=0.95)

    plt.suptitle("Posterior Distributions with 95% HDIs", fontsize=14)

    # Optional: Trace plots to visually inspect convergence
    az.plot_trace(idata, var_names=[
        "mu_d_prime", "mu_criterion",
        "beta_stimulus_d_prime", "beta_difficulty_d_prime",
        "beta_stimulus_criterion", "beta_difficulty_criterion"
    ])

    plt.suptitle("Trace Plots")

    # Optional: Forest plot
    az.plot_forest(idata, var_names=[
        "beta_stimulus_d_prime", "beta_difficulty_d_prime"
    ], combined=True)    

    plt.title("Forest Plot of Effects (Stimulus vs. Difficulty)")

    # Energy plot to diagnose sampling quality
    az.plot_energy(idata)
    plt.title("Energy Plot")

    # Step 5: Delta plots (already doing this — nice!)
    delta_data = read_data('data.csv', prepare_for='delta plots', display=True)
    # Define condition order and names for plotting
    condition_order = [0, 1, 2, 3]
    CONDITION_NAMES = {
        0: "Easy-Audio",
        1: "Hard-Audio",
        2: "Easy-Visual",
        3: "Hard-Visual"
    }

    # Draw delta plots
    draw_delta_plots(delta_data, pnum=1)

    plt.tight_layout()
    plt.show()
