#!/usr/bin/env python3
"""
Plot LOG2 test results from random_cases.csv

Usage:
    python plot_results.py [options]

Options:
    --input FILE        Input CSV file (default: build/random_cases.csv)
    --output FILE       Output image file (default: build/log2_plot.png)
    --plot CURVES       Comma-separated list of curves to plot
                        Options: dut, cpu_ref, gpu_ref
                        Examples: dut,cpu_ref  or  dut  or  cpu_ref,gpu_ref
                        Default: dut,cpu_ref
    --sample N          Plot only every N-th point (default: 100)
    --help              Show this help message

Examples:
    # Plot DUT and CPU reference
    python plot_results.py --plot dut,cpu_ref

    # Plot only DUT
    python plot_results.py --plot dut

    # Plot all available curves
    python plot_results.py --plot dut,cpu_ref,gpu_ref

    # Plot with custom sampling
    python plot_results.py --plot dut,cpu_ref --sample 1000
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Plot LOG2 test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', default='build/random_cases.csv',
                        help='Input CSV file (default: build/random_cases.csv)')
    parser.add_argument('--output', default='build/log2_plot.png',
                        help='Output image file (default: build/log2_plot.png)')
    parser.add_argument('--plot', default='dut,cpu_ref',
                        help='Comma-separated curves to plot (default: dut,cpu_ref)')
    parser.add_argument('--sample', type=int, default=100,
                        help='Plot every N-th point (default: 100)')

    args = parser.parse_args()

    # Read CSV file
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} data points from {args.input}")
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        print("Please run the simulation first to generate the data file")
        sys.exit(1)

    # Parse which curves to plot
    curves = [c.strip() for c in args.plot.split(',')]

    # Validate curve names
    available_columns = df.columns.tolist()
    available_curves = [col for col in available_columns if col != 'in']

    for curve in curves:
        if curve not in available_curves:
            print(f"Error: '{curve}' is not available in the data")
            print(f"Available curves: {', '.join(available_curves)}")
            sys.exit(1)

    # Sample data for plotting
    df_sampled = df.iloc[::args.sample]
    print(f"Plotting {len(df_sampled)} sampled points (every {args.sample}-th point)")

    # Create plot
    plt.figure(figsize=(12, 8))

    # Define colors and markers for different curves
    style_map = {
        'dut': {'color': 'blue', 'marker': 'o', 'label': 'DUT', 'alpha': 0.6},
        'cpu_ref': {'color': 'red', 'marker': 's', 'label': 'CPU Reference', 'alpha': 0.6},
        'gpu_ref': {'color': 'green', 'marker': '^', 'label': 'GPU Reference', 'alpha': 0.6}
    }

    # Plot each curve
    for curve in curves:
        style = style_map.get(curve, {'color': 'black', 'marker': 'x', 'alpha': 0.6})
        plt.scatter(df_sampled['in'], df_sampled[curve],
                   color=style['color'],
                   marker=style['marker'],
                   s=20,
                   alpha=style['alpha'],
                   label=style.get('label', curve))

    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('log2(x)', fontsize=12)
    plt.title(f'LOG2 Function - {", ".join([style_map.get(c, {}).get("label", c) for c in curves])}',
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)

    # Add reference log2 curve
    x_ref = np.linspace(df['in'].min(), df['in'].max(), 1000)
    y_ref = np.log2(x_ref)
    plt.plot(x_ref, y_ref, 'k--', alpha=0.3, linewidth=1, label='Ideal log2(x)')
    plt.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}")


if __name__ == '__main__':
    main()
