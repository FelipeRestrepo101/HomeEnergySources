import matplotlib.pyplot as plt
import seaborn as sns

def plot_energy_totals(df):
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available\nPlease select valid dates", ha='center', va='center')
        ax.set_axis_off()
        return fig

    df['value'] = df['value'].astype(int)
    sum_df = df.groupby('type-name').sum(numeric_only=True)
    plt.figure(figsize=(8, 8))
    graph = sns.barplot(sum_df, x='type-name', y='value', palette='flare', hue='type-name')
    graph.set_title("Energy Totals")
    graph.set_xlabel("Power Source")
    graph.set_ylabel("Count")
    plt.xticks(rotation=90)
    return graph

def plot_source_trend(df, source_choice, smooth_window):
    df = df[df['type-name'] == source_choice].copy()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available\nPlease select valid dates", ha='center', va='center')
        ax.set_axis_off()
        return fig

    df['value'] = df['value'].astype(int)
    df['period'] = pd.to_datetime(df['period'])
    df['value'] = df['value'].rolling(window=smooth_window, center=True, min_periods=1).mean()
    plt.figure(figsize=(8, 8))
    graph = sns.lineplot(df, x='period', y='value')
    plt.xticks(rotation=90)
    return graph