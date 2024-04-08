The cases presented in this repository are examples developed during the courses of Data Science

Here are the explanation regarding these cases:
1) Simple_Linear_Regression_sklearn.ipynb
   * Data - Real Estate (Price vs Size)
   * Libraries - Numpy, Pandas, Matplotlib, Scipy, Seaborn, SKLearn
   * Calculus of R-Squared, Coeficient and Interception
   * Ploting data at a scatter plot and the regression line
   
2) Example of T-Test
   * Data - Students Grades vs Attendance to Remedial Classes
   * Libraries (Pandas, Scipy)
   * Data splitted in Students that Attend Remedial Classes and did not Attend Remedial Classes

   * H0 = no significant difference in grades between students in remedial classes and theis peers
   * H1 = substantial discrepancy between the two groups
   * P-value = 0.115
   * p_value > 0.05, indicates we cannot reject H0
   * Assumptions:
   * Independence: Observations in each group are independent of each other. One student grade does not influence the grade of another
   * Normality: The data in each group is approximately normally distributed..
   * Equal or unequal variance: The t-test can be conducted under the assumption of equal variances between the two groups or not.
   * If not, a variation of the t-test called Welch's t-test can be used

   * Performing Levene's test for equal variances check
   * H0 = The variances of the two groups are equal
   * H1 = The variances of the two groups are not equal
   * P-value = 0.58
   * p_value > 0.05, fail to reject H0

   * Variance for Students that attended remedial classes = 75.87
   * Variance for Students that did not attended remedial classes = 82.63
   
3) EDA on Dataset (Exploratory Data Analysis
   * Data - Properties Location, Price and Size
   * Libraries - Pandas, Matplotlib
   * Plotting data in graphics and evaluate correlation between variables, outliers

4) Multiple Linear Regression
   * Data - SAT, GPA and a Random variable
   * Libraries - Numpy, Pandas, Matplotlib, Seaborn, Statsmodel, Scykit Learn
   * Calculus of R-Squared, Coeficients and Interception
   * R-Squared = 0.41
   * Coeficients = SAT (0.00165), Random (-0.00827)
   * Interception = 0.296
   
   * Calculus or Adjusted R-Squared (It penalize the excessive use of variables)
   * Adj. R² = 1-(1-R²) * (n-1) / (n-p-1)
   * Adj. R² = 0.3920
   * Adj. R² < R²
   
   * The OLS analysis indicates the following:
   * P-value is too high > 0.05
   * Although R-squared increases with variable Rand 1,2,3, the Adj R-squared decreases
   * R2 = 0.406 - without Rand1,2,3
   * R2 = 0.407 - with Rand 1,2,3
   * Adj R2 = 0.399 - without Rand1,2,3
   * Adj R2 = 0.392 - with Rand1,2,3 (Penalizes the usage of more variables that had no explanatory power)

5) 
