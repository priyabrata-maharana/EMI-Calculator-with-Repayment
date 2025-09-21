import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np



def emi(P, annual_rate, N):
    """Bank EMI formula"""
    r = annual_rate / 12.0
    if r == 0:
        return P / N
    return P * r * (1 + r)**N / ((1 + r)**N - 1)

def sip_future_value(monthly_invest, annual_rate, months):
    """SIP future value"""
    r = annual_rate / 12.0
    if r == 0:
        return monthly_invest * months
    return monthly_invest * ((1 + r)**months - 1) / r

def generate_amort_table(P, rL, EMI, N, prepay=0, interval=0):
    """Generate amortization table """
    r = rL / 12.0
    balance = P
    data = []
    cum_interest = 0
    cum_principal = 0
    
    for month in range(1, N+1):
        interest = balance * r
        principal = EMI - interest
        cum_interest += interest
        cum_principal += principal
        balance -= principal
        
        prepay_this = 0
        if interval > 0 and month % interval == 0:
            prepay_this = min(prepay, balance)
            balance -= prepay_this
            cum_principal += prepay_this
        
        balance = max(0, balance)
        
        
        data.append({
            'Month': month,
            'EMI': EMI,
            'Principal': principal,
            'Interest': interest,
            'Prepayment': prepay_this,
            'Balance': balance,
            'Cum. Principal': cum_principal,
            'Cum. Interest': cum_interest
        })
        
        if balance <= 0:
            break
    
    return pd.DataFrame(data)



st.set_page_config(
    page_title="EMI Calculator", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🏠 EMI Calculator & Investment Planner")
st.markdown("Calculate EMIs, compare with SIP investments, and analyze prepayment scenarios")

# Sidebar for inputs
st.sidebar.header("Loan Details")
price = st.sidebar.number_input("Home Price (₹)", min_value=0.0, value=10000000.0, step=100000.0)
cash = st.sidebar.number_input("Downpayment (₹)", min_value=0.0, value=5000000.0, step=100000.0)
loan_rate = st.sidebar.number_input("Loan Rate (%)", min_value=0.0, value=8.5, step=0.1) / 100
tenure = st.sidebar.number_input("Tenure (months)", min_value=1, value=240, step=12)

st.sidebar.header("Budget & Investment")
budget = st.sidebar.number_input("Monthly Budget (₹)", min_value=0.0, value=100000.0, step=5000.0)
sip_rate = st.sidebar.number_input("SIP Return Rate (%)", min_value=0.0, value=12.0, step=0.5) / 100

st.sidebar.header("Prepayment (Optional)")
prepay = st.sidebar.number_input("Prepayment Amount (₹)", min_value=0.0, value=100000.0, step=50000.0)
interval = st.sidebar.number_input("Prepayment Interval (months)", min_value=0, value=12, step=6)

# Calculate loan amount
P = max(price - cash, 0.0)

if P <= 0:
    st.error("❌ No loan required - cash covers full price!")
    st.stop()

# Calculate EMI
EMI = emi(P, loan_rate, tenure)

if EMI > budget:
    st.error(f"❌ EMI ₹{EMI:,.0f} exceeds your monthly budget of ₹{budget:,.0f}")
    st.stop()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📋 Amortization", "🎨 Charts", "💡 Scenarios"])

with tab1:
    st.header("Loan Summary")
    
    # Basic metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Loan Amount", f"₹{P:,.0f}")
        st.metric("Monthly EMI", f"₹{EMI:,.0f}")
        st.metric("Tenure", f"{tenure} months ({tenure/12:.1f} years)")
    
    with col2:
        total_payment = EMI * tenure
        total_interest = total_payment - P
        st.metric("Total Payment", f"₹{total_payment:,.0f}")
        st.metric("Total Interest", f"₹{total_interest:,.0f}")
        interest_pct = (total_interest / total_payment) * 100
        st.metric("Interest % of Total", f"{interest_pct:.1f}%")
    
    # SIP Calculation
    sip_invest = max(budget - EMI, 0)
    sip_value = sip_future_value(sip_invest, sip_rate, tenure)
    
    st.subheader("Investment Summary")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("SIP Monthly", f"₹{sip_invest:,.0f}")
    with col4:
        st.metric("SIP Future Value", f"₹{sip_value:,.0f}")
    with col5:
        total_wealth = sip_value + P  # Home value + SIP corpus
        st.metric("Total Wealth", f"₹{total_wealth:,.0f}")
    
    # Progress indicator
    st.progress(tenure / tenure)
    st.caption(f"Loan progress: {tenure} of {tenure} months")

with tab2:
    st.header("📋 Amortization Schedule")
    
    # Generate amortization table
    df = generate_amort_table(P, loan_rate, EMI, tenure, prepay, interval)
    
    # Summary stats
    total_principal = df['Cum. Principal'].iloc[-1]
    total_interest_paid = df['Cum. Interest'].iloc[-1]
    actual_tenure = len(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actual Tenure", f"{actual_tenure} months")
    with col2:
        st.metric("Total Principal", f"₹{total_principal:,.0f}")
    with col3:
        st.metric("Total Interest", f"₹{total_interest_paid:,.0f}")
    
    # Format DataFrame for display
    df_display = df.copy()
   
    currency_cols = ['EMI', 'Principal', 'Interest', 'Prepayment', 'Cum. Principal', 'Cum. Interest']
    for col in currency_cols:
        df_display[col] = df_display[col].apply(lambda x: f"₹{x:,.0f}")
    
    # Format balance
    df_display['Balance'] = df_display['Balance'].apply(lambda x: f"₹{x:,.0f}")
    
    # Display table with pagination
    st.dataframe(df_display, use_container_width=True, height=400)
    
    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"amortization_schedule_{actual_tenure}months.csv",
        mime="text/csv"
    )

with tab3:
    st.header("🎨 Visualizations")
    
    total_payment = EMI * tenure
    total_interest = total_payment - P
    
    # Pie chart - Principal vs Interest
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie([P, total_interest], 
                labels=['Principal', 'Interest'], 
                autopct='%1.1f%%', 
                colors=['#66b3ff', '#ff9999'], 
                startangle=90,
                textprops={'fontsize': 10})
        ax1.set_title('Principal vs Interest Breakdown', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig1)
    
    # Loan Balance vs SIP Growth
    with col2:
        # Generate balance series
        balances = [P]
        balance = P
        r_monthly = loan_rate / 12.0
        
        for month in range(1, tenure + 1):
            interest = balance * r_monthly
            principal = EMI - interest
            balance = max(balance - principal, 0)
            balances.append(balance)
        
        # Generate SIP values
        months = list(range(tenure + 1))
        sip_values = [sip_future_value(sip_invest, sip_rate, i) for i in months]
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(months, balances, 'b-', label='Loan Balance', linewidth=2, marker='o', markersize=3)
        ax2.plot(months, sip_values, 'g-', label='SIP Value', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Amount (₹)', fontsize=12)
        ax2.set_title('Loan Balance vs SIP Growth Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

with tab4:
    st.header("💡 Prepayment Scenarios Analysis")
    
    
    total_payment = EMI * tenure
    original_interest = total_payment - P
    
    if prepay > 0 and interval > 0:
        st.info(f"🔔 Analyzing prepayments of ₹{prepay:,.0f} every {interval} months")
        
        # Scenario 1: Keep EMI same, reduce tenure
        balance1 = P
        total_interest1 = 0
        total_paid1 = 0
        month1 = 0
        r_monthly = loan_rate / 12
        
        prepayment_count1 = 0
        while balance1 > 0 and month1 < tenure * 2:
            month1 += 1
            interest = balance1 * r_monthly
            total_interest1 += interest
            principal = EMI - interest
            total_paid1 += EMI
            balance1 -= principal
            
            # Apply prepayment
            if month1 % interval == 0 and balance1 > 0:
                prepay_amount = min(prepay, balance1)
                balance1 -= prepay_amount
                total_paid1 += prepay_amount
                prepayment_count1 += 1
            
            balance1 = max(0, balance1)
            if balance1 <= 0:
                # Adjust final payment if overpaid
                if balance1 < 0:
                    total_paid1 += balance1
                    total_interest1 += balance1
                break
        
        months_saved1 = max(0, tenure - month1)
        total_principal1 = total_paid1 - total_interest1
        
        # Scenario 2: Reduce EMI, keep tenure same
        balance2 = P
        total_interest2 = 0
        total_paid2 = 0
        current_emi = EMI
        emi_history = []
        prepayment_count2 = 0
        
        for month in range(1, tenure + 1):
            interest = balance2 * r_monthly
            total_interest2 += interest
            principal = current_emi - interest
            total_paid2 += current_emi
            balance2 -= principal
            
            # Apply prepayment and recalculate EMI
            if month % interval == 0 and balance2 > 0:
                prepay_amount = min(prepay, balance2)
                balance2 -= prepay_amount
                total_paid2 += prepay_amount
                prepayment_count2 += 1
                balance2 = max(0, balance2)
                
                remaining_months = tenure - month
                if remaining_months > 0 and balance2 > 0:
                    current_emi = emi(balance2, loan_rate, remaining_months)
                    emi_history.append((month, current_emi))
            
            if balance2 <= 0:
                # Adjust final payment if overpaid
                if balance2 < 0:
                    total_paid2 += balance2
                    total_interest2 += balance2
                break
        
        total_principal2 = total_paid2 - total_interest2
        
        # Display comparison metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Plan", f"₹{original_interest:,.0f}")
        with col2:
            st.metric("Scenario 1: Reduce Tenure", f"₹{total_interest1:,.0f}")
            st.metric("Months Saved", f"{months_saved1}")
        with col3:
            st.metric("Scenario 2: Reduce EMI", f"₹{total_interest2:,.0f}")
            final_emi = current_emi if emi_history else EMI
            st.metric("Final EMI", f"₹{final_emi:,.0f}")
        
        # Interest savings
        st.subheader("💰 Interest Savings")
        col1, col2 = st.columns(2)
        with col1:
            savings1 = original_interest - total_interest1
            st.success(f"**Reduce Tenure:** ₹{savings1:,.0f} saved")
            st.info(f"Prepayments made: {prepayment_count1}")
        with col2:
            savings2 = original_interest - total_interest2
            st.success(f"**Reduce EMI:** ₹{savings2:,.0f} saved")
            st.info(f"Prepayments made: {prepayment_count2}")
        
        # PIE CHARTS FOR PREPAYMENT SCENARIOS
        st.subheader("📊 Interest vs Principal Breakdown")
        
        # Create pie charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Scenario 1 Pie Chart
            fig_s1, ax_s1 = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax_s1.pie([total_principal1, total_interest1], 
                                                labels=['Principal + Prepayments', 'Interest'], 
                                                autopct='%1.1f%%', 
                                                colors=['#66b3ff', '#ff9999'], 
                                                startangle=90,
                                                textprops={'fontsize': 10})
            ax_s1.set_title(f'Scenario 1: Reduce Tenure\n(Savings: ₹{savings1:,.0f})', 
                           fontsize=12, fontweight='bold')
            
            
            ax_s1.text(0, 0, f'Total: ₹{total_paid1:,.0f}\nInterest: ₹{total_interest1:,.0f}', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax_s1.transAxes, fontsize=10, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig_s1)
        
        with col2:
            # Scenario 2 Pie Chart
            fig_s2, ax_s2 = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax_s2.pie([total_principal2, total_interest2], 
                                                labels=['Principal + Prepayments', 'Interest'], 
                                                autopct='%1.1f%%', 
                                                colors=['#66b3ff', '#ff9999'], 
                                                startangle=90,
                                                textprops={'fontsize': 10})
            ax_s2.set_title(f'Scenario 2: Reduce EMI\n(Savings: ₹{savings2:,.0f})', 
                           fontsize=12, fontweight='bold')
            
            # Add total values as text
            ax_s2.text(0, 0, f'Total: ₹{total_paid2:,.0f}\nInterest: ₹{total_interest2:,.0f}', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax_s2.transAxes, fontsize=10, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig_s2)
        
        # Comparison table
        st.subheader("📈 Scenario Comparison")
        comparison_data = {
            'Metric': ['Total Interest', 'Total Principal', 'Total Payment', 'Interest %', 'Months/Years'],
            'Original': [
                f"₹{original_interest:,.0f}",
                f"₹{P:,.0f}",
                f"₹{total_payment:,.0f}",
                f"{(original_interest/total_payment)*100:.1f}%",
                f"{tenure} months"
            ],
            'Scenario 1 (Reduce Tenure)': [
                f"₹{total_interest1:,.0f}",
                f"₹{total_principal1:,.0f}",
                f"₹{total_paid1:,.0f}",
                f"{(total_interest1/total_paid1)*100:.1f}%",
                f"{month1} months ({month1/12:.1f} yrs)"
            ],
            'Scenario 2 (Reduce EMI)': [
                f"₹{total_interest2:,.0f}",
                f"₹{total_principal2:,.0f}",
                f"₹{total_paid2:,.0f}",
                f"{(total_interest2/total_paid2)*100:.1f}%",
                f"{tenure} months ({tenure/12:.1f} yrs)"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # EMI history chart for Scenario 2
        if emi_history:
            st.subheader("📈 EMI Changes Over Time (Scenario 2)")
            
            
            all_emi_changes = [(0, EMI)]  # Starting point
            all_emi_changes.extend(emi_history)
            
            emi_months = [change[0] for change in all_emi_changes]
            emi_values = [change[1] for change in all_emi_changes]
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(emi_months, emi_values, 'ro-', linewidth=2, markersize=6, color='#ff6b6b')
            ax3.set_xlabel('Month', fontsize=12)
            ax3.set_ylabel('EMI (₹)', fontsize=12)
            ax3.set_title('EMI Reduction Over Time - Scenario 2', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            
            for i, (month, emi_val) in enumerate(zip(emi_months, emi_values)):
                if i % 2 == 0:  
                    ax3.annotate(f'₹{emi_val:,.0f}', (month, emi_val), 
                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Show recent EMI changes
            st.write("**EMI Change History:**")
            changes_df = pd.DataFrame(emi_history, columns=['Month', 'New EMI'])
            changes_df['New EMI'] = changes_df['New EMI'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(changes_df, use_container_width=True)
            
    else:
        st.info("👆 Enter prepayment details above to analyze scenarios")
        
        # Original pie chart for reference
        st.subheader("📊 Original Plan - Interest vs Principal")
        fig_orig, ax_orig = plt.subplots(figsize=(8, 8))
        ax_orig.pie([P, original_interest], 
                   labels=['Principal', 'Interest'], 
                   autopct='%1.1f%%', 
                   colors=['#66b3ff', '#ff9999'], 
                   startangle=90,
                   textprops={'fontsize': 12})
        ax_orig.set_title('Original Loan Plan - Principal vs Interest', fontsize=16, fontweight='bold')
        
        # Add total values
        ax_orig.text(0, 0, f'Total Payment: ₹{total_payment:,.0f}\nInterest: ₹{original_interest:,.0f}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax_orig.transAxes, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        
        plt.tight_layout()
        st.pyplot(fig_orig)
        
        st.markdown("""
        ### How Prepayment Works
        **Scenario 1: Reduce Tenure** 🏃‍♂️
        - Keep same EMI payment
        - Loan gets paid off **faster**
        - **Maximum interest savings**
        - Best for people who want to be debt-free quickly
        
        **Scenario 2: Reduce EMI** 💸
        - Keep same loan tenure
        - **Monthly payments decrease**
        - Better monthly cash flow
        - Good for people who need lower EMIs
        
        **The Pie Charts Show:**
        - **Blue slice**: Principal + Prepayments (what you get back)
        - **Red slice**: Interest (what you lose to the bank)
        - **Smaller red slice = Better deal!**
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    *Built by Priyabrata Maharana (priyabrata.maharana1236@gmail.com) | All calculations are approximate and for educational purposes only. 
    Please consult your financial advisor before making investment decisions.*
    """
)

# Add some CSS styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .stMetric > label {
        color: #1f77b4;
        font-weight: bold;
    }
</style>

""", unsafe_allow_html=True)
