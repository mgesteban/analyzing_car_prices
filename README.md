# Used Car Price Analysis

A comprehensive data science project analyzing factors that drive used car prices to provide actionable insights for used car dealerships.

## About This Project

This project represents my capstone work for the Professional Certificate in Machine Learning and Artificial Intelligence at UC Berkeley College of Engineering in California. I analyzed 10,000 of the 426,880 used car records to identify key price drivers and provide data-driven recommendations for inventory acquisition and pricing strategies. Using machine learning models, I achieved ~$8,500 RMSE for Ridge Regression. However, after running the full dataset, RMSE for Lasso is only $7, 426.  

Since I have studied Linear, Ridge, and Lasso Regression, I decided to focus on these three models to strengthen my skills and gain a deeper intuition about how they perform in practice. To build confidence, I first ran experiments on a 10,000-row sample before scaling up to the full dataset of 426,880 records. One of my biggest challenges was managing the missing values (missingness). This required me to revisit concepts like one-hot encoding and carefully apply them so that categorical variables could be transformed into numerical features the models could use.

After cleaning and preparing the dataset, I compared the three models using cross-validation. The results showed very similar errors, but Lasso Regression slightly outperformed the others on this run:

=== CROSS-VALIDATION COMPARISON (lower RMSE is better) ===
Model              CV_RMSE        Best_Params
0   Lasso Regression   7,533.49   {'regressor__alpha': 0.1}
1   Linear Regression  7,533.58   {}
2   Ridge Regression   7,533.70   {'regressor__alpha': 1}

=== BEST MODEL: Lasso Regression ===


### Key Findings

- **Car age and mileage** are the strongest price predictors
- **Premium manufacturers** command significant price premiums  
- **Vehicle condition** and transmission type significantly impact pricing
- **Depreciation rate**: ~$2,000-3,000 per year on average

## Project Structure

```
analyzing_car_prices/
├── README.md                           # Project documentation
├── data/
│   └── vehicles.csv                   # Original dataset (426K records)
├── Report.md                          # Detailed analysis report
├── analyzing_car_prices.ipynb         # Complete analysis notebook
├── requirements.txt                   # Python dependencies
├── images/                            # Generated charts and visualizations
└── LICENSE                           # Project license
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd analyzing_car_prices
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook analyzing_car_prices.ipynb
   ```

## Results Summary

### Model Performance
- **Best Model**: Ridge Regression
- **Test RMSE**: ~$8,500
- **Test R²**: 0.66
- **Cross-validation**: 3-fold CV with hyperparameter tuning

### Top Price Drivers
Based on permutation importance analysis:
1. **Car Age** - Most significant factor
2. **Odometer Reading** - Strong negative correlation
3. **Manufacturer** - Premium brands maintain value
4. **Condition** - Excellent condition commands premium
5. **Transmission Type** - Automatic vs manual preference

### Business Impact Analysis

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| **Age** | -$2,500/year avg | Focus on 3-7 year old vehicles |
| **Mileage** | Significant drop >100k | Prioritize <60k mile vehicles |
| **Condition** | 20-40% price variance | Emphasize excellent condition |
| **Brand** | Premium +30-50% | Target high-value manufacturers |

## Business Recommendations

### For Used Car Dealerships

#### Inventory Acquisition Strategy
- **Focus on newer vehicles** (3-7 years old) for optimal profit margins
- **Prioritize low-mileage vehicles** (under 60,000 miles) when possible
- **Target premium manufacturers** that maintain resale value
- **Avoid high-mileage vehicles** unless priced significantly below market

#### Pricing Strategy
- **Use age as primary pricing factor** - expect $2,000-3,000 depreciation per year
- **Adjust pricing based on mileage brackets**:
  - 0-30k miles: Premium pricing
  - 30-60k miles: Standard pricing  
  - 60-100k miles: Moderate discount
  - 100k+ miles: Significant discount
- **Premium pricing for excellent condition** vehicles
- **Consider transmission preferences** in your local market

#### Market Positioning
- **Highlight low mileage** as key selling point in marketing
- **Emphasize vehicle condition** in all marketing materials
- **Leverage brand reputation** - premium manufacturers justify higher prices
- **Be transparent about age impact** on pricing

## Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

1. **Business Understanding**: Define objectives for used car dealership
2. **Data Understanding**: Explore 426K vehicle records, identify quality issues
3. **Data Preparation**: Clean data, engineer features, handle missing values
4. **Modeling**: Compare multiple ML algorithms, cross-validate, tune hyperparameters
5. **Evaluation**: Assess model performance, extract feature importance
6. **Deployment**: Generate actionable business recommendations

### Data Processing Pipeline

```python
# Data cleaning steps
1. Remove missing prices (target variable)
2. Filter unrealistic prices ($500-$100,000)
3. Clean year data (1990-2023)
4. Clean odometer data (0-500,000 miles)
5. Handle missing categorical data
6. Engineer new features (car_age, manufacturer_grouped)
7. Prepare for modeling (scaling, encoding)
```

### Model Comparison

| Model | CV RMSE | Test RMSE | R² Score |
|-------|---------|-----------|----------|
| Ridge Regression | $8,400 | $8,500 | 0.66 |
| Linear Regression | $8,600 | $8,700 | 0.64 |
| Lasso Regression | $8,700 | $8,800 | 0.63 |

## Data Source

- **Dataset**: Kaggle Used Cars Dataset (subset of 3M records)
- **Size**: 426,880 records
- **Features**: 18 columns including price, year, manufacturer, model, condition, etc.
- **Target**: Price (continuous variable, $500-$100,000 range)

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **jupyter**: Interactive development environment

### Key Features
- **Modular code structure** for reusability
- **Comprehensive data validation** and cleaning
- **Cross-validation** with hyperparameter tuning
- **Feature importance analysis** for interpretability
- **Business-focused insights** and recommendations

## Academic Context

This project was completed as part of my Professional Certificate in Machine Learning and Artificial Intelligence at UC Berkeley College of Engineering in California. It demonstrates practical application of:

- **Data Science Methodology** (CRISP-DM framework)
- **Machine Learning Algorithms** (regression, ensemble methods)
- **Feature Engineering** and data preprocessing
- **Model Evaluation** and cross-validation techniques
- **Business Intelligence** and actionable insights generation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or collaboration opportunities, please reach out through the repository issues or contact information.

---

**Note**: This analysis provides data-driven insights for business decision-making. Always consider market conditions, regional variations, and business constraints when implementing recommendations.
