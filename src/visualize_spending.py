import os
import json
import pandas as pd
from datetime import datetime

# File paths
INPUT_JSON = "../outputs/transactions_array.json"
OUTPUT_HTML = "../outputs/spending_dashboard.html"

def load_transactions():
    """Load transaction data from JSON file"""
    # Check if the file exists
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found!")
        print("Please run extract_template.py first")
        return None
    
    # Read the JSON file
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        transactions = json.load(f)
    
    print(f"Loaded {len(transactions)} transactions")
    return transactions

def prepare_data(transactions):
    """Convert transactions to DataFrame and add date columns"""
    df = pd.DataFrame(transactions)
    
    # Convert to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Add month columns for grouping
    df['month_year'] = df['transaction_date'].dt.strftime('%Y-%m')
    df['month_name'] = df['transaction_date'].dt.strftime('%B %Y')
    
    return df

def get_monthly_stats(df, month_year, transaction_type):
    """Calculate statistics for a specific month and type (charge or payment)"""
    # Filter to specific month and type
    month_df = df[(df['month_year'] == month_year) & 
                  (df['transaction_type'] == transaction_type)].copy()
    
    # Return None if no data
    if len(month_df) == 0:
        return None
    
    # Build list of transactions
    transactions_list = []
    for _, row in month_df.iterrows():
        txn = {
            'transaction_date': row['transaction_date'].strftime('%Y-%m-%d'),
            'transaction_description': row['transaction_description'],
            'spend_category': row['spend_category'],
            'amount': float(row['amount'])
        }
        transactions_list.append(txn)
    
    # Calculate statistics
    stats = {
        'total_spending': float(month_df['amount'].sum()),
        'num_transactions': int(len(month_df)),
        'avg_transaction': float(month_df['amount'].mean()),
        'category_breakdown': {k: float(v) for k, v in month_df.groupby('spend_category')['amount'].sum().to_dict().items()},
        'daily_spending': {str(k): float(v) for k, v in month_df.groupby(month_df['transaction_date'].dt.day)['amount'].sum().to_dict().items()},
        'transactions': transactions_list
    }
    
    return stats

def generate_html_dashboard(df):
    """Create the HTML dashboard with embedded JavaScript"""
    # Get all unique months
    months = sorted(df['month_year'].unique())
    
    # Process data for both charges and payments
    charges_data = {}
    payments_data = {}
    
    for month in months:
        charges = get_monthly_stats(df, month, 'charge')
        payments = get_monthly_stats(df, month, 'payment')
        
        if charges:
            charges_data[month] = charges
        if payments:
            payments_data[month] = payments
    
    # Build month selector options
    month_options = []
    for month in months:
        month_name = df[df['month_year'] == month]['month_name'].iloc[0]
        month_options.append(f'<option value="{month}">{month_name}</option>')
    
    # Create the HTML content with embedded CSS and JavaScript
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Dashboard - Payments & Charges</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}
        
        .control-row {{
            display: flex;
            gap: 20px;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .control-row:last-child {{
            margin-bottom: 0;
        }}
        
        .control-row label {{
            font-size: 1.1em;
            font-weight: 600;
            color: #333;
            min-width: 100px;
        }}
        
        .control-row select {{
            flex: 1;
            padding: 12px 20px;
            font-size: 1.1em;
            border: 2px solid #667eea;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .control-row select:hover {{
            border-color: #764ba2;
        }}
        
        .view-tabs {{
            display: flex;
            gap: 10px;
            flex: 1;
        }}
        
        .view-tab {{
            flex: 1;
            padding: 12px 20px;
            font-size: 1.1em;
            font-weight: 600;
            border: 2px solid #667eea;
            border-radius: 8px;
            background: white;
            color: #667eea;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }}
        
        .view-tab:hover {{
            background: #f0f4ff;
        }}
        
        .view-tab.active {{
            background: #667eea;
            color: white;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .stat-value {{
            color: #333;
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .stat-value.payment {{
            color: #27ae60;
        }}
        
        .stat-value.charge {{
            color: #e74c3c;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .chart-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .chart-title {{
            color: #333;
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
        }}
        
        .table-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .amount {{
            font-weight: 600;
        }}
        
        .amount.charge {{
            color: #e74c3c;
        }}
        
        .amount.payment {{
            color: #27ae60;
        }}
        
        .category-badge {{
            display: inline-block;
            padding: 4px 12px;
            background: #667eea;
            color: white;
            border-radius: 20px;
            font-size: 0.85em;
        }}
        
        .no-data {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
            color: #666;
            font-size: 1.2em;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            h1 {{
                font-size: 1.8em;
            }}
            
            .control-row {{
                flex-direction: column;
                align-items: stretch;
            }}
            
            .control-row label {{
                min-width: auto;
            }}
            
            .view-tabs {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💳 Credit Card Dashboard</h1>
            <p class="subtitle">Separate view for payments and charges</p>
        </div>
        
        <div class="controls">
            <div class="control-row">
                <label for="monthSelect">📅 Month:</label>
                <select id="monthSelect" onchange="updateDashboard()">
                    {''.join(month_options)}
                </select>
            </div>
            <div class="control-row">
                <label>📊 View:</label>
                <div class="view-tabs">
                    <button class="view-tab active" onclick="switchView('charges')" id="chargesTab">
                        💸 Charges
                    </button>
                    <button class="view-tab" onclick="switchView('payments')" id="paymentsTab">
                        💰 Payments
                    </button>
                </div>
            </div>
        </div>
        
        <div id="dashboard"></div>
    </div>
    
    <script>
        // Embed the data directly in JavaScript
        const chargesData = {json.dumps(charges_data)};
        const paymentsData = {json.dumps(payments_data)};
        let currentView = 'charges';
        let categoryChart = null;
        let dailyChart = null;
        
        function switchView(view) {{
            currentView = view;
            
            // Toggle tab highlighting
            document.getElementById('chargesTab').classList.remove('active');
            document.getElementById('paymentsTab').classList.remove('active');
            
            if (view === 'charges') {{
                document.getElementById('chargesTab').classList.add('active');
            }} else {{
                document.getElementById('paymentsTab').classList.add('active');
            }}
            
            // Refresh the dashboard
            updateDashboard();
        }}
        
        function updateDashboard() {{
            const selectedMonth = document.getElementById('monthSelect').value;
            const data = currentView === 'charges' ? chargesData[selectedMonth] : paymentsData[selectedMonth];
            
            // Handle no data case
            if (!data) {{
                document.getElementById('dashboard').innerHTML = `
                    <div class="no-data">
                        <p>📭 No ${{currentView}} data available for this month</p>
                    </div>
                `;
                return;
            }}
            
            const viewLabel = currentView === 'charges' ? 'Charges' : 'Payments';
            const amountClass = currentView === 'charges' ? 'charge' : 'payment';
            
            // Build the dashboard HTML
            document.getElementById('dashboard').innerHTML = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total ${{viewLabel}}</div>
                        <div class="stat-value ${{amountClass}}">$${{data.total_spending.toFixed(2)}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Number of ${{viewLabel}}</div>
                        <div class="stat-value">${{data.num_transactions}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average ${{viewLabel.slice(0, -1)}}</div>
                        <div class="stat-value ${{amountClass}}">$${{data.avg_transaction.toFixed(2)}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Categories</div>
                        <div class="stat-value">${{Object.keys(data.category_breakdown).length}}</div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    <div class="chart-card">
                        <h3 class="chart-title">${{viewLabel}} by Category</h3>
                        <canvas id="categoryChart"></canvas>
                    </div>
                    <div class="chart-card">
                        <h3 class="chart-title">Daily ${{viewLabel}} Trend</h3>
                        <canvas id="dailyChart"></canvas>
                    </div>
                </div>
                
                <div class="table-card">
                    <h3 class="chart-title">${{viewLabel}} Details</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Description</th>
                                <th>Category</th>
                                <th>Amount</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${{data.transactions.map(t => `
                                <tr>
                                    <td>${{t.transaction_date}}</td>
                                    <td>${{t.transaction_description}}</td>
                                    <td><span class="category-badge">${{t.spend_category}}</span></td>
                                    <td class="amount ${{amountClass}}">$${{t.amount.toFixed(2)}}</td>
                                </tr>
                            `).join('')}}
                        </tbody>
                        <tfoot>
                            <tr>
                                <td colspan="3" style="text-align: right; font-weight: bold;">Total:</td>
                                <td class="amount ${{amountClass}}" style="font-size: 1.2em;">$${{data.total_spending.toFixed(2)}}</td>
                            </tr>
                        </tfoot>
                    </table>
                </div>
            `;
            
            // Create the charts
            createCategoryChart(data.category_breakdown);
            createDailyChart(data.daily_spending);
        }}
        
        function createCategoryChart(categoryData) {{
            const ctx = document.getElementById('categoryChart').getContext('2d');
            
            // Destroy old chart if it exists
            if (categoryChart) {{
                categoryChart.destroy();
            }}
            
            const categories = Object.keys(categoryData);
            const amounts = Object.values(categoryData);
            
            // Create pie chart
            categoryChart = new Chart(ctx, {{
                type: 'pie',
                data: {{
                    labels: categories,
                    datasets: [{{
                        data: amounts,
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                            '#9966FF', '#FF9F40', '#FF6384', '#36A2EB'
                        ]
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return context.label + ': $' + context.parsed.toFixed(2);
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        function createDailyChart(dailyData) {{
            const ctx = document.getElementById('dailyChart').getContext('2d');
            
            // Destroy old chart if it exists
            if (dailyChart) {{
                dailyChart.destroy();
            }}
            
            // Sort days numerically
            const days = Object.keys(dailyData).sort((a, b) => parseInt(a) - parseInt(b));
            const amounts = days.map(day => dailyData[day]);
            
            // Set colors based on view
            const color = currentView === 'charges' ? '#e74c3c' : '#27ae60';
            const bgColor = currentView === 'charges' ? 'rgba(231, 76, 60, 0.1)' : 'rgba(39, 174, 96, 0.1)';
            
            // Create line chart
            dailyChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: days.map(d => 'Day ' + d),
                    datasets: [{{
                        label: currentView === 'charges' ? 'Charges ($)' : 'Payments ($)',
                        data: amounts,
                        borderColor: color,
                        backgroundColor: bgColor,
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{
                                callback: function(value) {{
                                    return '$' + value;
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Initialize dashboard on page load
        updateDashboard();
    </script>
</body>
</html>"""
    
    return html_content

def main():
    print("="*60)
    print("Credit Card Dashboard Generator")
    print("Separate Payments & Charges View")
    print("="*60 + "\n")
    
    # Load the transactions
    transactions = load_transactions()
    if not transactions:
        return
    
    # Prepare the data
    df = prepare_data(transactions)
    
    # Show some basic statistics
    charges = df[df['transaction_type'] == 'charge']
    payments = df[df['transaction_type'] == 'payment']
    
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Total Charges: {len(charges)} (${charges['amount'].sum():,.2f})")
    print(f"Total Payments: {len(payments)} (${payments['amount'].sum():,.2f})")
    print(f"Months available: {len(df['month_year'].unique())}\n")
    
    html_content = generate_html_dashboard(df)
    
    # Make sure output directory exists
    os.makedirs("../outputs", exist_ok=True)
    
    # Save the HTML file
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Dashboard saved to: {OUTPUT_HTML}")
    print(f"✓ Features:")
    print(f"  - Separate tabs for Charges and Payments")
    print(f"  - Month selector")
    print(f"  - Category breakdown charts")
    print(f"  - Daily spending trends")
    print(f"  - Detailed transaction tables")
    print(f"\n✓ Open the file in your browser\n")
    print("="*60)

if __name__ == "__main__":
    main()