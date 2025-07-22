# 🎮 League of Legends Confidence Intervals Learning Tool

An interactive web application for teaching confidence intervals using real League of Legends game duration data.

**Created by Professor Cynthia McGinnis**  
*Introduction to Statistics Course*

##  **Access the App**

**Live Web App:** [Your Streamlit URL will go here]

##  **What This Tool Teaches**

- **Confidence Intervals**: Understand how to calculate and interpret confidence intervals
- **Sample vs Population**: See the relationship between samples and populations
- **Coverage Rates**: Test how often confidence intervals actually contain the true parameter
- **Statistical Distributions**: Learn when to use t-distribution vs z-distribution

##  **Features**

- **Real Data**: Uses statistics from 2,060 actual LoL ranked games
- **Interactive Sampling**: Draw new samples and see how confidence intervals change
- **Multiple Confidence Levels**: Compare 80%, 90%, 95%, and 99% confidence intervals
- **Coverage Simulation**: Run simulations to verify theoretical coverage rates
- **Visual Learning**: Side-by-side population and sample distribution plots

##  **Dataset Information**

- **Population Size**: 2,060 LoL ranked games
- **Population Mean (μ)**: 30.54 minutes
- **Population Standard Deviation (σ)**: 8.53 minutes
- **Distribution**: Approximately normal with realistic bounds (15-65 minutes)

##  **For Instructors: Running Locally**

If you want to run this locally or modify it:

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/[your-username]/lol-confidence-intervals.git
cd lol-confidence-intervals

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run lol_confidence_intervals.py
```

The app will open in your browser at `http://localhost:8501`

##  **Educational Use**

This tool is designed for statistics courses covering:
- Descriptive Statistics
- Probability Distributions
- Confidence Intervals
- Central Limit Theorem
- Hypothesis Testing

Perfect for:
- **Lecture Demonstrations**: Show confidence intervals in real-time
- **Lab Activities**: Students explore different sample sizes and confidence levels
- **Homework Assignments**: Students can run simulations and report findings
- **Class Discussions**: Compare theoretical vs observed coverage rates

##  **Learning Objectives**

After using this tool, students will be able to:
1. Calculate confidence intervals for population means
2. Interpret confidence interval results in context
3. Understand the relationship between confidence level and interval width
4. Explain what "95% confident" actually means
5. Recognize factors that affect confidence interval precision

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

##  **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 **Contact**

Professor Cynthia McGinnis  
cynthia.mcginnis@umgc.edu
UMGC

---

*Made with ❤️ for statistics education*
