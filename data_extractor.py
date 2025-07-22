#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
"""
Data Extractor for LoL Game Duration
This script reads your Excel file and creates the student app with embedded data

Run this ONCE to create the student version
"""

import pandas as pd
import numpy as np

def extract_game_duration_data():
    """
    Extract game duration data from your Excel file and create the student app
    """
    file_path = "/Users/cynthiamcginnis/Documents/statistics/LoLgames.xlsx"
    
    try:
        # Load your data
        print("📁 Loading data from your file...")
        full_data = pd.read_excel(file_path)
        print(f"✅ Loaded {len(full_data)} games")
        print(f"📊 Available columns: {list(full_data.columns)}")
        
        # Find game duration column
        duration_col = None
        for col in full_data.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            if 'gameduration' in col_lower or 'duration' in col_lower:
                duration_col = col
                break
        
        if duration_col:
            print(f"🎯 Found duration column: {duration_col}")
            
            # Extract and convert to minutes
            duration_seconds = full_data[duration_col].dropna()
            duration_minutes = duration_seconds / 60.0
            
            # Clean data (remove outliers)
            duration_minutes = duration_minutes[(duration_minutes >= 10) & (duration_minutes <= 80)]
            
            print(f"📈 Converted {len(duration_minutes)} games to minutes")
            print(f"📊 Range: {duration_minutes.min():.1f} to {duration_minutes.max():.1f} minutes")
            print(f"📊 Mean: {duration_minutes.mean():.2f} minutes")
            
            # Convert to list for embedding in code
            duration_list = duration_minutes.tolist()
            
            return duration_list, duration_col
        else:
            print("❌ Could not find game duration column")
            return None, None
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None, None

def create_student_app(duration_data, original_col_name):
    """
    Create the student version of the app with embedded data
    """
    
    app_code = f'''#!/usr/bin/env python3
"""
League of Legends Game Duration Confidence Intervals - Student Version
Interactive web application for understanding confidence intervals

Author: Professor Cynthia McGinnis
Course: Introduction to Statistics

Run with: streamlit run lol_confidence_intervals_student.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import logging
import os

# Suppress warnings and logs for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'
plt.switch_backend('Agg')

class GameDurationConfidenceAnalyzer:
    """
    A class to analyze League of Legends game duration data and teach confidence intervals
    """
    
    def __init__(self):
        """Initialize with embedded game duration data"""
        # Real LoL game duration data (converted from seconds to minutes)
        # Original column: {original_col_name}
        self.game_durations = {duration_data}
        
        self.population_data = pd.Series(self.game_durations)
        self.population_mean = self.population_data.mean()
        self.population_std = self.population_data.std()
        
        print(f"📊 Loaded {{len(self.population_data)}} real LoL games")
        print(f"📈 Population mean: {{self.population_mean:.2f}} minutes")
        print(f"📈 Population std: {{self.population_std:.2f}} minutes")
    
    def draw_sample(self, sample_size=30):
        """Draw a random sample from the population"""
        if sample_size > len(self.population_data):
            sample_size = len(self.population_data)
        
        sample = self.population_data.sample(n=sample_size, random_state=np.random.randint(1000))
        return sample
    
    def calculate_confidence_interval(self, sample_data, confidence_level=0.95):
        """Calculate confidence interval for game duration"""
        n = len(sample_data)
        sample_mean = sample_data.mean()
        sample_std = sample_data.std(ddof=1)
        
        # Determine whether to use z or t distribution
        use_t_dist = n < 30
        alpha = 1 - confidence_level
        
        if use_t_dist:
            df = n - 1
            critical_value = stats.t.ppf(1 - alpha/2, df)
            distribution_used = f"t-distribution (df={{df}})"
        else:
            critical_value = stats.norm.ppf(1 - alpha/2)
            distribution_used = "z-distribution"
        
        # Calculate margin of error (E) and confidence interval
        standard_error = sample_std / np.sqrt(n)
        margin_of_error = critical_value * standard_error
        
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error
        
        # Check if CI contains population parameter
        contains_population = lower_bound <= self.population_mean <= upper_bound
        
        return {{
            'sample_size': n,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'confidence_level': confidence_level,
            'distribution': distribution_used,
            'critical_value': critical_value,
            'standard_error': standard_error,
            'margin_of_error': margin_of_error,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'contains_population': contains_population
        }}
    
    def create_visualization(self, sample_data, confidence_level=0.95):
        """Create visualization showing population and sample distributions"""
        result = self.calculate_confidence_interval(sample_data, confidence_level)
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create smooth x-axis for normal curves
        x_min = min(self.population_data.min(), sample_data.min()) - 5
        x_max = max(self.population_data.max(), sample_data.max()) + 5
        x_smooth = np.linspace(x_min, x_max, 300)
        
        # Plot 1: Population Distribution
        ax1.hist(self.population_data, bins=40, density=True, alpha=0.7, 
                color='lightblue', edgecolor='black', label='Population Data')
        
        # Population normal curve
        pop_normal = stats.norm(self.population_mean, self.population_std)
        ax1.plot(x_smooth, pop_normal.pdf(x_smooth), 'b-', linewidth=3, label='Normal Curve')
        
        # Population mean line
        ax1.axvline(self.population_mean, color='red', linestyle='-', linewidth=3, 
                   label=f'μ = {{self.population_mean:.1f}} min')
        
        ax1.set_xlabel('Game Duration (minutes)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'Population Distribution\\n(N = {{len(self.population_data)}} games)', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample Distribution
        ax2.hist(sample_data, bins=15, density=True, alpha=0.7, 
                color='lightgreen', edgecolor='black', label='Sample Data')
        
        # Sample normal curve
        sample_normal = stats.norm(result['sample_mean'], result['sample_std'])
        ax2.plot(x_smooth, sample_normal.pdf(x_smooth), 'g-', linewidth=3, label='Normal Curve')
        
        # Sample mean line
        ax2.axvline(result['sample_mean'], color='darkgreen', linestyle='-', linewidth=3, 
                   label=f'x̄ = {{result["sample_mean"]:.1f}} min')
        
        # Population mean for comparison
        ax2.axvline(self.population_mean, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Population μ = {{self.population_mean:.1f}} min')
        
        # Confidence interval bounds
        ax2.axvline(result['lower_bound'], color='purple', linestyle=':', linewidth=3, 
                   label=f'{{confidence_level*100}}% CI Bounds')
        ax2.axvline(result['upper_bound'], color='purple', linestyle=':', linewidth=3)
        
        ax2.set_xlabel('Game Duration (minutes)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(f'Sample Distribution\\n(n = {{len(sample_data)}} games)', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, result

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="LoL Confidence Intervals",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and header
    st.title("🎮 League of Legends Game Duration")
    st.title("📊 Confidence Intervals Learning Tool")
    
    st.markdown("""
    **Created by Professor Cynthia McGinnis**  
    *Introduction to Statistics*  
    *GitHub: https://github.com/cynthialmcginnis*
    """)
    
    st.markdown("---")
    
    # Initialize analyzer with embedded data
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GameDurationConfidenceAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    st.sidebar.subheader("🔬 Analysis Settings")
    
    sample_size = st.sidebar.slider(
        "Sample Size (n):",
        min_value=10,
        max_value=min(200, len(analyzer.population_data)),
        value=30,
        step=5
    )
    
    confidence_level = st.sidebar.selectbox(
        "Confidence Level:",
        [0.80, 0.90, 0.95, 0.99],
        index=2,
        format_func=lambda x: f"{{x*100:.0f}}%"
    )
    
    # Action buttons
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("🎯 Analyze New Sample", type="primary")
    compare_button = st.sidebar.button("📊 Compare Confidence Levels")
    simulate_button = st.sidebar.button("🎲 Run Coverage Simulation")
    
    # Show population information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Population Size", f"{{len(analyzer.population_data):,}} games")
    with col2:
        st.metric("Population Mean (μ)", f"{{analyzer.population_mean:.2f}} min")
    with col3:
        st.metric("Population Std (σ)", f"{{analyzer.population_std:.2f}} min")
    with col4:
        st.metric("Range", f"{{analyzer.population_data.min():.1f}} - {{analyzer.population_data.max():.1f}} min")
    
    # Analysis sections
    if analyze_button or 'current_sample' not in st.session_state:
        st.session_state.current_sample = analyzer.draw_sample(sample_size)
    
    if 'current_sample' in st.session_state:
        sample = st.session_state.current_sample
        
        # Create visualization
        fig, result = analyzer.create_visualization(sample, confidence_level)
        
        # Display visualization
        st.subheader("📊 Population vs Sample Distributions")
        st.pyplot(fig)
        
        # Display results in organized columns
        st.subheader("🎯 Confidence Interval Analysis Results")
        
        # Sample statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Sample Statistics")
            st.metric("Sample Size (n)", result['sample_size'])
            st.metric("Sample Mean (x̄)", f"{{result['sample_mean']:.2f}} min")
            st.metric("Sample Std (s)", f"{{result['sample_std']:.2f}} min")
            st.metric("Distribution Used", result['distribution'])
        
        with col2:
            st.markdown("### 🔢 Critical Values & Error")
            st.metric("Critical Value", f"{{result['critical_value']:.3f}}")
            st.metric("Standard Error", f"{{result['standard_error']:.3f}} min")
            st.metric("Margin of Error (E)", f"{{result['margin_of_error']:.3f}} min")
            st.metric("Confidence Level", f"{{result['confidence_level']*100:.0f}}%")
        
        # Confidence interval results
        st.markdown("### 📊 Confidence Interval")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Lower Bound", f"{{result['lower_bound']:.2f}} min")
        with col2:
            st.metric("Upper Bound", f"{{result['upper_bound']:.2f}} min")
        with col3:
            interval_width = result['upper_bound'] - result['lower_bound']
            st.metric("Interval Width", f"{{interval_width:.2f}} min")
        
        # Interpretation
        st.markdown("### 💡 Statistical Interpretation")
        st.info(f"""
        **We are {{result['confidence_level']*100:.0f}}% confident that the true population mean game duration 
        is between {{result['lower_bound']:.2f}} and {{result['upper_bound']:.2f}} minutes.**
        """)
        
        # Population parameter check
        st.markdown("### 🔍 Population Parameter Verification")
        if result['contains_population']:
            st.success(f"""
            ✅ **SUCCESS!** The confidence interval CONTAINS the population parameter!
            
            - True Population Mean (μ): {{analyzer.population_mean:.2f}} minutes
            - Confidence Interval: [{{result['lower_bound']:.2f}}, {{result['upper_bound']:.2f}}] minutes
            - This interval correctly captures the true population mean.
            """)
        else:
            miss_rate = (1 - result['confidence_level']) * 100
            st.error(f"""
            ❌ **MISS!** The confidence interval DOES NOT contain the population parameter.
            
            - True Population Mean (μ): {{analyzer.population_mean:.2f}} minutes
            - Confidence Interval: [{{result['lower_bound']:.2f}}, {{result['upper_bound']:.2f}}] minutes
            - This represents the ~{{miss_rate:.0f}}% chance that our interval fails to capture μ.
            """)
    
    # Handle comparison button
    if compare_button:
        st.markdown("---")
        st.subheader("📊 Confidence Level Comparison")
        
        sample = analyzer.draw_sample(sample_size)
        confidence_levels = [0.80, 0.90, 0.95, 0.99]
        
        comparison_data = []
        for cl in confidence_levels:
            result = analyzer.calculate_confidence_interval(sample, cl)
            comparison_data.append({{
                'Confidence Level': f"{{cl*100:.0f}}%",
                'Lower Bound': f"{{result['lower_bound']:.2f}}",
                'Upper Bound': f"{{result['upper_bound']:.2f}}",
                'Margin of Error (E)': f"{{result['margin_of_error']:.2f}}",
                'Contains μ': "✅" if result['contains_population'] else "❌"
            }})
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        st.info(f"Population Mean (μ) = {{analyzer.population_mean:.2f}} minutes")
        st.markdown("**Notice:** Higher confidence levels create wider intervals!")
    
    # Handle simulation button
    if simulate_button:
        st.markdown("---")
        st.subheader("🎲 Coverage Rate Simulation")
        
        n_trials = 100
        successes = 0
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_trials):
            sample = analyzer.population_data.sample(n=sample_size, replace=True)
            result = analyzer.calculate_confidence_interval(sample, confidence_level)
            if result['contains_population']:
                successes += 1
            
            # Update progress
            progress_bar.progress((i + 1) / n_trials)
            status_text.text(f"Running simulation: {{i + 1}}/{{n_trials}}")
        
        coverage_rate = successes / n_trials
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trials Run", n_trials)
        with col2:
            st.metric("Intervals Containing μ", successes)
        with col3:
            st.metric("Observed Coverage", f"{{coverage_rate*100:.1f}}%")
        with col4:
            st.metric("Expected Coverage", f"{{confidence_level*100:.0f}}%")
        
        if abs(coverage_rate - confidence_level) < 0.05:
            st.success("✅ Coverage rate matches expectation!")
        else:
            st.warning("⚠️ Coverage rate differs from expectation (normal with small samples)")

if __name__ == "__main__":
    main()
'''
    
    return app_code

def main():
    """Main function to extract data and create student app"""
    print("🎮 LoL Confidence Intervals - Data Extractor")
    print("=" * 50)
    
    # Extract data from your file
    duration_data, col_name = extract_game_duration_data()
    
    if duration_data:
        print(f"✅ Successfully extracted {len(duration_data)} game durations")
        
        # Create student app code
        print("📝 Creating student app...")
        app_code = create_student_app(duration_data, col_name)
        
        # Save student app
        with open("lol_confidence_intervals_student.py", "w") as f:
            f.write(app_code)
        
        print("✅ Created: lol_confidence_intervals_student.py")
        print("\n🎯 Next Steps:")
        print("1. Give students: lol_confidence_intervals_student.py")
        print("2. Give students: requirements.txt")  
        print("3. Students run: pip install -r requirements.txt")
        print("4. Students run: streamlit run lol_confidence_intervals_student.py")
        print("\n📊 Students will have real LoL data embedded in their app!")
        
    else:
        print("❌ Could not extract data. Check your file path.")

if __name__ == "__main__":
    main()


# In[ ]:




