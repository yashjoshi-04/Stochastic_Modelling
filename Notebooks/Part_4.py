import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

def setup_plot_style():
    """Set up the plotting style for consistent visualizations"""
    pass

def plot_hedging_process(T, process_data, steps):
    """Plot the hedging process using Plotly"""
    days = T * 365
    
    black = "#000000"
    gold = "#FFD700"
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Stock Price', 'Delta', 'Stock Holding', 
                       'Bond Holding', 'Hedged Portfolio', 'Hedging Error'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Define plot configurations
    
    plots = [
        ('Stock Price', process_data['stock_price'], 1, 1),
        ('Delta', process_data['delta'], 1, 2),
        ('Stock Holding', process_data['stock_holding'], 2, 1),
        ('Bond Holding', process_data['bond_holding'], 2, 2),
        ('Hedged Portfolio', process_data['hedged_portfolio'], 3, 1),
    ]
    
    # Create individual plots
    
    for title, data, row, col in plots:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=data,
                mode='lines+markers' if steps <= 60 else 'lines',
                marker=dict(
                    symbol='circle',
                    size=4,
                    color=black
                ),
                line=dict(color=black, width=1),
                name=title,
                showlegend=False
            ),
            row=row, col=col
        )
    
    # Add hedging error plots
    
    fig.add_trace(
        go.Scatter(
            x=days,
            y=process_data['stock_hedge_error'],
            mode='lines+markers' if steps <= 60 else 'lines',
            marker=dict(
                symbol='circle',
                size=4,
                color=black
            ),
            line=dict(color=black, width=1),
            name='Stock Hedge Error'
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=days,
            y=process_data['bond_hedge_error'],
            mode='lines+markers' if steps <= 60 else 'lines',
            marker=dict(
                symbol='circle',
                size=4,
                color=gold
            ),
            line=dict(color=gold, width=1),
            name='Bond Hedge Error'
        ),
        row=3, col=2
    )

    fig.update_layout(
        title=dict(
            text=f"Black-Scholes Hedging Process (N = {steps} steps)",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=14, weight='bold')
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.0,
            borderwidth=1
        ),
        height=900,
        width=1200,
        paper_bgcolor='white',  
        plot_bgcolor='white'    
    )

    # Update all subplots to have white background
    
    fig.update_xaxes(
        title_text="Days", 
        gridcolor='lightgrey', 
        gridwidth=1, 
        griddash='dash',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        gridcolor='lightgrey', 
        gridwidth=1, 
        griddash='dash',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    return fig

def plot_pl_distribution(pl_results):
    """Plot P&L distribution using Plotly"""
    if not pl_results:
        return None
    
    num_plots = len(pl_results)
    purple = '#800080' 
    
    fig = make_subplots(
        rows=1, cols=num_plots,
        subplot_titles=[f'Histogram (N = {steps}) Rebalancing Steps' 
                       for steps, _, _ in pl_results]
    )
    
    for i, (steps, final_PL, stats) in enumerate(pl_results):
        if len(final_PL) == 0:
            print(f"No data to plot for step {steps}.")
            continue
            
        hist_data = np.histogram(
            final_PL,
            bins=30,
            range=(-1.5, 1.5),
            weights=np.ones(len(final_PL)) / len(final_PL) * 100
        )
        
        fig.add_trace(
            go.Bar(
                x=[(hist_data[1][i] + hist_data[1][i+1])/2 for i in range(len(hist_data[1])-1)],
                y=hist_data[0],
                marker_color=purple, 
                opacity=1.0,
                name=f'N={steps}'
            ),
            row=1, col=i+1
        )
        
        stats_text = (
            f"μ = {stats['mean']:.2f}<br>"
            f"σ = {stats['std']:.2f}<br>"
            f"SD/Premium = {stats['sd_premium']:.2f}"
        )
        
        fig.add_annotation(
            x=1.3,
            y=25,
            xref=f"x{i+1}",
            yref=f"y{i+1}",
            text=stats_text,
            showarrow=False,
            bgcolor="white",
            bordercolor="gold",
            borderwidth=1,
            font=dict(size=12)
        )

    fig.update_layout(
        showlegend=False,
        height=500,
        width=1200,
        bargap=0.1,
        paper_bgcolor='white',  
        plot_bgcolor='white'    
    )
    
    fig.update_xaxes(
        title_text='Final Profit/Loss',
        range=[-1.5, 1.5],
        dtick=0.5,
        minor=dict(ticklen=3, tickcolor="gold", tickwidth=1),
        gridcolor='lightgrey',
        gridwidth=1,
        griddash='dash',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    fig.update_yaxes(
        title_text='Frequency (out of 100%)',
        gridcolor='lightgrey',
        gridwidth=1,
        griddash='dash',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True
    )
    
    return fig


def black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def phi(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def psi_Bt(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return -K*np.exp(-r*T)*norm.cdf(d2)

def simulate_brownian_motion(paths, steps, T):
    deltaT = T/steps
    t = np.linspace(0, T, steps+1)
    X = np.c_[np.zeros((paths, 1)), np.random.randn(paths, steps)]
    return t, np.cumsum(np.sqrt(deltaT) * X, axis=1)

def calculate_hedging_error(S0, K, r, sigma, T, t, x):
    S = pd.DataFrame(S0 * np.exp((r-sigma**2/2)*t + sigma*x))
    phi_values = pd.DataFrame(phi(S, K, r, sigma, T-t))
    psib_values = pd.DataFrame(psi_Bt(S, K, r, sigma, T-t))
    hedging_error = pd.DataFrame(
        phi_values.values[:,:-1] * np.diff(S) +
        psib_values.values[:,:-1] * r * (T/len(t))
    )
    hedging_error['Sum'] = hedging_error.apply(np.sum, axis=1)
    payoff = np.maximum(S-K, 0)
    call_option = black_scholes_call(S0, K, r, sigma, T)
    final_PL = hedging_error['Sum'].values + call_option - payoff.iloc[:,-1]
    return final_PL

def main():
    r, S0, K = 0.05, 100, 100
    sigma = 0.2
    maturity = 1/12
    paths = 50000
    list_N = [21, 84]
    pl_results = []
    
    for steps in list_N:
        T, W_T = simulate_brownian_motion(paths, steps, maturity)
        
        dt = maturity / steps
        blackscholespath = S0*np.exp((r-sigma**2/2)*T + sigma*W_T[0])
        
        process_data = {
            'stock_price': blackscholespath,
            'delta': [],
            'stock_holding': [],
            'bond_holding': [],
            'hedged_portfolio': [],
            'stock_hedge_error': [0],
            'bond_hedge_error': [0]
        }
        
        prev_phi = prev_bond_pos = None
        for t, S_t in zip(T, blackscholespath):
            delta = phi(S_t, K, r, sigma, maturity-t)
            stock_pos = delta * S_t
            bond_pos = psi_Bt(S_t, K, r, sigma, maturity-t)
            process_data['delta'].append(delta)
            process_data['stock_holding'].append(stock_pos)
            process_data['bond_holding'].append(bond_pos)
            process_data['hedged_portfolio'].append(stock_pos - bond_pos)
            if prev_phi is not None:
                process_data['stock_hedge_error'].append(prev_phi*S_t - stock_pos)
                process_data['bond_hedge_error'].append(prev_bond_pos*np.exp(r*dt) - bond_pos)
            prev_phi = delta
            prev_bond_pos = bond_pos
            
        fig_process = plot_hedging_process(T, process_data, steps)
        fig_process.show()
        fig_process.write_image(f"hedging_process_N{steps}.png")  
        
        final_PL = calculate_hedging_error(S0, K, r, sigma, maturity, T, W_T)
        
        stats = {
            'mean': round(np.mean(final_PL), 2),
            'std': round(np.std(final_PL), 2),
            'sd_premium': round(np.std(final_PL)/black_scholes_call(S0, K, r, sigma, maturity), 2)*100
        }
        
        print(f"\nResults for N = {steps}:")
        print(f"Mean P&L: {stats['mean']}")
        print(f"Standard Deviation: {stats['std']}")
        print(f"SD as % of option premium: {stats['sd_premium']}%")
        
        pl_results.append((steps, final_PL, stats))
    
    fig_pl = plot_pl_distribution(pl_results)
    fig_pl.show()
    fig_pl.write_image("pl_distribution.png")  

if __name__ == "__main__":
    main()
    
