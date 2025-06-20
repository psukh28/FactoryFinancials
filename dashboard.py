import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title='Maesot Factory Dashboard', layout='wide')

# --- Load Data ---
df = pd.read_csv('Cleaned_Monthly_Data.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

df_totals = pd.read_csv('maesot - Totals.csv')
df_analysis = pd.read_csv('maesot - Analysis.csv')

# --- Derived Fields for Monthly ---
df['output_per_worker'] = df['output_pcs_/month'] / df['cost_per_worker']
df['cost_per_output'] = df['total_cost_of_running_maesot_factory'] / df['output_pcs_/month']
df['overhead_cost'] = df['total_salary_per_month'] + df['kk_support_workers']
df['utility_cost'] = df['electricity'] + df['rent'] + df['water']
df['admin_cost'] = df['transport'] + df['tax'] + df['social']
df['overhead_pct'] = 100 * df['overhead_cost'] / df['total_cost_of_running_maesot_factory']
df['production_cost'] = df['total_labour_charges_per_month']

# --- Tabs ---
tabs = st.tabs(["Monthly Dashboard", "Yearly Overview", "Yearly Analysis"])

# --- Monthly Dashboard Tab ---
with tabs[0]:
    # --- Sidebar Filters ---
    years = sorted(df['year'].unique())
    selected_year = st.sidebar.selectbox('Select Year', years, index=len(years)-1)
    df_year = df[df['year'] == selected_year]

    # Month Filter UI
    st.sidebar.markdown('#### Month Filter')
    if len(df_year) > 1:
        min_date = df_year['date'].min().date()
        max_date = df_year['date'].max().date()
        date_range = st.sidebar.slider(
            "Select Month Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MMM YYYY"
        )
        # Convert slider output (date) back to datetime for filtering
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
        df_filtered = df_year[(df_year['date'] >= start_dt) & (df_year['date'] <= end_dt)]
        # Show selected range as caption
        start_label = start_dt.strftime('%b %Y')
        end_label = end_dt.strftime('%b %Y')
        st.sidebar.caption(f"Showing: {start_label} to {end_label}")
    else:
        df_filtered = df_year.copy()
        st.sidebar.caption(f"Showing: {df_year['date'].iloc[0].strftime('%b %Y')}")

    # --- KPI Calculations ---
    def kpi_delta(curr, prev):
        if prev == 0 or pd.isna(prev):
            return None
        return (curr - prev) / prev * 100

    # --- Month Selector for KPIs ---
    if not df_filtered.empty:
        month_display_options = [f"{row['month_id']} ({row['month']})" for _, row in df_filtered.iterrows()]
        default_idx = len(df_filtered) - 1
        selected_kpi_idx = st.selectbox('Select Month for KPIs', options=range(len(df_filtered)), format_func=lambda i: month_display_options[i], index=default_idx)
        kpi_row = df_filtered.iloc[selected_kpi_idx]
        if selected_kpi_idx > 0:
            kpi_prev_row = df_filtered.iloc[selected_kpi_idx - 1]
        else:
            kpi_prev_row = None
    else:
        kpi_row = kpi_prev_row = None

    # KPIs for selected month
    if kpi_row is not None:
        kpi_output = int(kpi_row['output_pcs_/month'])
        kpi_cost = int(kpi_row['total_cost_of_running_maesot_factory'])
        kpi_cost_per_output = kpi_row['cost_per_output']
        kpi_output_per_worker = kpi_row['output_per_worker']
        kpi_overhead_pct = kpi_row['overhead_pct']
        # Deltas
        if kpi_prev_row is not None:
            delta_output = kpi_delta(kpi_row['output_pcs_/month'], kpi_prev_row['output_pcs_/month'])
            delta_cost = kpi_delta(kpi_row['total_cost_of_running_maesot_factory'], kpi_prev_row['total_cost_of_running_maesot_factory'])
            delta_cost_per_output = kpi_delta(kpi_row['cost_per_output'], kpi_prev_row['cost_per_output'])
            delta_output_per_worker = kpi_delta(kpi_row['output_per_worker'], kpi_prev_row['output_per_worker'])
            delta_overhead_pct = kpi_delta(kpi_row['overhead_pct'], kpi_prev_row['overhead_pct'])
        else:
            delta_output = delta_cost = delta_cost_per_output = delta_output_per_worker = delta_overhead_pct = None

        # --- KPI Row ---
        with st.container():
            st.markdown(f"### Key Performance Indicators for {kpi_row['month_id']} ({kpi_row['month']})")
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            # 1. Total Output: Higher is better (green up, red down)
            with kpi1:
                st.markdown(f"**Total Output**")
                st.markdown(f"<span style='font-size:2em;font-weight:bold'>{kpi_output:,}</span>", unsafe_allow_html=True)
                if delta_output is not None:
                    arrow = '▲' if delta_output >= 0 else '▼'
                    color = 'green' if delta_output >= 0 else 'red'
                    st.markdown(f"<span style='color:{color}'> {arrow} {delta_output:.1f}% vs prev. month</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:gray'>--</span> <span style='color:gray'>N/A (first month)</span>", unsafe_allow_html=True)
                st.caption('Higher is better')
            # 2. Factory Running Cost: Lower is better (green down, red up)
            with kpi2:
                st.markdown(f"**Factory Running Cost (THB)**")
                st.markdown(f"<span style='font-size:2em;font-weight:bold'>{kpi_cost:,}</span>", unsafe_allow_html=True)
                if delta_cost is not None:
                    arrow = '▼' if delta_cost <= 0 else '▲'
                    color = 'green' if delta_cost <= 0 else 'red'
                    st.markdown(f"<span style='color:{color}'> {arrow} {abs(delta_cost):.1f}% vs prev. month</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:gray'>--</span> <span style='color:gray'>N/A (first month)</span>", unsafe_allow_html=True)
                st.caption('Lower is better')
            # 3. Cost per Output: Lower is better (green down, red up)
            with kpi3:
                st.markdown(f"**Cost per Output (THB/pc)**")
                st.markdown(f"<span style='font-size:2em;font-weight:bold'>{kpi_cost_per_output:,.2f}</span>", unsafe_allow_html=True)
                if delta_cost_per_output is not None:
                    arrow = '▼' if delta_cost_per_output <= 0 else '▲'
                    color = 'green' if delta_cost_per_output <= 0 else 'red'
                    st.markdown(f"<span style='color:{color}'> {arrow} {abs(delta_cost_per_output):.1f}% vs prev. month</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:gray'>--</span> <span style='color:gray'>N/A (first month)</span>", unsafe_allow_html=True)
                st.caption('Lower is better')
            # 4. Output per Worker: Higher is better (green up, red down)
            with kpi4:
                st.markdown(f"**Output per Worker**")
                st.markdown(f"<span style='font-size:2em;font-weight:bold'>{kpi_output_per_worker:,.2f}</span>", unsafe_allow_html=True)
                if delta_output_per_worker is not None:
                    arrow = '▲' if delta_output_per_worker >= 0 else '▼'
                    color = 'green' if delta_output_per_worker >= 0 else 'red'
                    st.markdown(f"<span style='color:{color}'> {arrow} {abs(delta_output_per_worker):.1f}% vs prev. month</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:gray'>--</span> <span style='color:gray'>N/A (first month)</span>", unsafe_allow_html=True)
                st.caption('Higher is better')
            # 5. Overhead % of Total Cost: Lower is better (green down, red up)
            with kpi5:
                st.markdown(f"**Overhead % of Total Cost**")
                st.markdown(f"<span style='font-size:2em;font-weight:bold'>{kpi_overhead_pct:.1f}%</span>", unsafe_allow_html=True)
                if delta_overhead_pct is not None:
                    arrow = '▼' if delta_overhead_pct <= 0 else '▲'
                    color = 'green' if delta_overhead_pct <= 0 else 'red'
                    st.markdown(f"<span style='color:{color}'> {arrow} {abs(delta_overhead_pct):.1f}% vs prev. month</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:gray'>--</span> <span style='color:gray'>N/A (first month)</span>", unsafe_allow_html=True)
                st.caption('Lower is better')
            st.markdown('---')

        # --- Delta Table in Expander ---
        with st.expander('📊 Show All Months: KPI & Delta Table'):
            table_data = []
            df_filtered_reset = df_filtered.reset_index(drop=True)
            for i, row in df_filtered_reset.iterrows():
                if i > 0:
                    prev = df_filtered_reset.iloc[i-1]
                    d_output = kpi_delta(row['output_pcs_/month'], prev['output_pcs_/month'])
                    d_cost = kpi_delta(row['total_cost_of_running_maesot_factory'], prev['total_cost_of_running_maesot_factory'])
                    d_cost_per_output = kpi_delta(row['cost_per_output'], prev['cost_per_output'])
                    d_output_per_worker = kpi_delta(row['output_per_worker'], prev['output_per_worker'])
                    d_overhead_pct = kpi_delta(row['overhead_pct'], prev['overhead_pct'])
                else:
                    d_output = d_cost = d_cost_per_output = d_output_per_worker = d_overhead_pct = None
                table_data.append({
                    'Month': f"{row['month_id']} ({row['month']})",
                    'Output': int(row['output_pcs_/month']),
                    'Output Δ%': f"{d_output:.1f}%" if d_output is not None else '--',
                    'Cost': int(row['total_cost_of_running_maesot_factory']),
                    'Cost Δ%': f"{d_cost:.1f}%" if d_cost is not None else '--',
                    'Cost/Output': f"{row['cost_per_output']:.2f}",
                    'Cost/Output Δ%': f"{d_cost_per_output:.1f}%" if d_cost_per_output is not None else '--',
                    'Output/Worker': f"{row['output_per_worker']:.2f}",
                    'Output/Worker Δ%': f"{d_output_per_worker:.1f}%" if d_output_per_worker is not None else '--',
                    'Overhead %': f"{row['overhead_pct']:.1f}%",
                    'Overhead % Δ': f"{d_overhead_pct:.1f}%" if d_overhead_pct is not None else '--',
                })
            st.dataframe(pd.DataFrame(table_data))
    else:
        st.info('No data to display for the selected year/month(s).')

    # --- Monthly Output vs. Factory Running Cost (Bar + Line, Dual Axis) ---
    if not df_filtered.empty:
        st.markdown('### Output vs. Factory Running Cost')
        df_plot = df_filtered.copy()
        # Smoothing (3-month rolling average)
        df_plot['cost_smooth'] = df_plot['total_cost_of_running_maesot_factory'].rolling(window=3, min_periods=1).mean()

        fig = go.Figure()
        # Output (left y-axis, bar)
        fig.add_trace(go.Bar(
            x=df_plot['date'], y=df_plot['output_pcs_/month'], name='Output (pcs)', marker_color='#4F81BD', yaxis='y1', hovertemplate='Output: %{y:,.0f} pcs'))
        # Cost (right y-axis, smoothed line)
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['cost_smooth'], name='Cost (3-mo avg)', yaxis='y2', line=dict(color='#C00000', width=3), hovertemplate='Cost (avg): %{y:,.0f} THB'))

        fig.update_layout(
            xaxis_title='Month',
            yaxis=dict(title='Output (pcs)', side='left', showgrid=False, zeroline=False),
            yaxis2=dict(title='Cost (THB)', side='right', overlaying='y', showgrid=False, zeroline=False),
            legend_title='Metric',
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=14),
            xaxis=dict(tickangle=-30),
            bargap=0.2,
            margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('---')

    # --- Cost per Output (Smoothed Line, Own Section) ---
    if not df_filtered.empty:
        st.markdown('### Cost per Output Over Time')
        df_plot = df_filtered.copy()
        df_plot['cost_per_output_smooth'] = df_plot['cost_per_output'].rolling(window=3, min_periods=1).mean()
        last_val = df_plot['cost_per_output'].iloc[-1]
        avg_val = df_plot['cost_per_output'].mean()
        delta = (last_val - avg_val) / avg_val * 100 if avg_val != 0 else 0
        st.info(f"📉 Latest cost per output is {last_val:.2f} THB — {'{:+.1f}'.format(delta)}% from the average of {avg_val:.2f} THB.")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['cost_per_output_smooth'],
            name='Cost per Output (3-mo avg)', line=dict(color='#F4B183', width=3),
            hovertemplate='Cost/Output: %{y:,.2f} THB/pc'))
        fig2.update_layout(
            xaxis_title='Month',
            yaxis_title='THB / piece',
            template='plotly_white',
            hovermode='x unified',
            font=dict(size=14),
            xaxis=dict(tickangle=-30)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('---')

    # --- Cost Breakdown Section ---
    if not df_filtered.empty:
        with st.container():
            st.markdown('### Monthly Cost Breakdown')
            cost_fig = go.Figure()
            cost_fig.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['production_cost'], name='Labor', marker_color='#4F81BD'))
            cost_fig.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['overhead_cost'], name='Overhead', marker_color='#A6A6A6'))
            cost_fig.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['utility_cost'], name='Utilities', marker_color='#9BC2E6'))
            cost_fig.add_trace(go.Bar(x=df_filtered['date'], y=df_filtered['admin_cost'], name='Admin', marker_color='#F4B183'))
            cost_fig.update_layout(
                barmode='stack',
                xaxis_title='Month',
                yaxis_title='THB',
                template='plotly_white',
                legend_title='Cost Type',
                hovermode='x unified',
                font=dict(size=14),
                xaxis=dict(tickangle=-30)
            )
            st.plotly_chart(cost_fig, use_container_width=True)
            st.markdown('---')
    else:
        st.info('No cost breakdown to display for the selected year/month(s).')

    # --- % Contribution Stacked Bar Section ---
    if not df_filtered.empty:
        st.markdown('### Cost Structure: % Contribution by Component')
        df_percent = df_filtered.copy()
        cost_total = df_percent[['production_cost', 'overhead_cost', 'utility_cost', 'admin_cost']].sum(axis=1)
        for col in ['production_cost', 'overhead_cost', 'utility_cost', 'admin_cost']:
            df_percent[col + '_pct'] = df_percent[col] / cost_total * 100
        percent_fig = go.Figure()
        percent_fig.add_trace(go.Bar(
            x=df_percent['date'],
            y=df_percent['production_cost_pct'],
            name='Labor',
            marker_color='#4F81BD',
            hovertemplate='%{y:.1f}%'))
        percent_fig.add_trace(go.Bar(
            x=df_percent['date'],
            y=df_percent['overhead_cost_pct'],
            name='Overhead',
            marker_color='#A6A6A6',
            hovertemplate='%{y:.1f}%'))
        percent_fig.add_trace(go.Bar(
            x=df_percent['date'],
            y=df_percent['utility_cost_pct'],
            name='Utilities',
            marker_color='#9BC2E6',
            hovertemplate='%{y:.1f}%'))
        percent_fig.add_trace(go.Bar(
            x=df_percent['date'],
            y=df_percent['admin_cost_pct'],
            name='Admin',
            marker_color='#F4B183',
            hovertemplate='%{y:.1f}%'))
        percent_fig.update_layout(
            barmode='stack',
            xaxis_title='Month',
            yaxis_title='% of Total Cost',
            yaxis=dict(range=[0, 100]),
            template='plotly_white',
            legend_title='Cost Type',
            hovermode='x unified',
            font=dict(size=14),
            xaxis=dict(tickangle=-30),
            showlegend=True
        )
        st.plotly_chart(percent_fig, use_container_width=True)
        st.markdown('---')

# --- Yearly Overview Tab ---
with tabs[1]:
    st.markdown('### Yearly KPIs & Trends')
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric('Output per Worker', f"{df_totals['Output pcs per worker'].iloc[-1]:,.2f}")
    with kpi_cols[1]:
        st.metric('Cost per Worker', f"{df_totals['Cost per worker'].iloc[-1]:,.2f}")
    with kpi_cols[2]:
        st.metric('Wages per Hr', f"{df_totals['Only Wages Per Hr'].iloc[-1]:,.2f}")
    with kpi_cols[3]:
        st.metric('Total Cost', f"{df_totals['Total Cost'].iloc[-1]:,.0f}")

    st.markdown('---')
    st.markdown('#### Yearly Trends')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_totals['Year'], y=df_totals['Output pcs /month'], name='Output pcs /month', marker_color='#4F81BD'))
    fig.add_trace(go.Scatter(x=df_totals['Year'], y=df_totals['Total Cost of  running Maesot factory'], name='Total Cost of Running Factory', line=dict(color='#C00000', width=3)))
    fig.add_trace(go.Scatter(x=df_totals['Year'], y=df_totals['Output pcs per worker'], name='Output per Worker', line=dict(color='#375A7F', width=3, dash='dot')))
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Value',
        legend_title='Metric',
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('---')
    st.markdown('#### Yearly Comparison Table')
    st.dataframe(df_totals)

# --- Yearly Analysis Tab ---
with tabs[2]:
    st.markdown('### Yearly Labor & Cost Analysis')
    st.dataframe(df_analysis)
    st.markdown('---')
    st.markdown('#### Key Metrics Over Time')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_analysis['Year'], y=df_analysis['Actual Paid'], name='Actual Paid', marker_color='#4F81BD'))
    fig.add_trace(go.Bar(x=df_analysis['Year'], y=df_analysis['Wage Per Day'], name='Wage Per Day', marker_color='#A6A6A6'))
    fig.add_trace(go.Scatter(x=df_analysis['Year'], y=df_analysis['Maesot running cost per hr'], name='Running Cost per Hr', line=dict(color='#C00000', width=3)))
    fig.add_trace(go.Scatter(x=df_analysis['Year'], y=df_analysis['Actual paid per worker per hour'], name='Actual Paid per Worker per Hr', line=dict(color='#F4B183', width=3, dash='dot')))
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='THB',
        legend_title='Metric',
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True) 