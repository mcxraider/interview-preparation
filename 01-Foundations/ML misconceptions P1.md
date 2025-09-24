# Machine Learning & Data Science Interview Prep Notes

## 1. L1 and L2 Regularization

### ELI5 Version
Imagine you're trying to fit a line through some points, but your line is getting too wiggly and complex. Regularization is like adding a "penalty" for making the line too complicated. L1 regularization forces some parts of the line to be exactly zero (like deleting unnecessary parts), while L2 regularization makes all parts smaller and smoother.

### Technical Deep Dive

#### L1 Regularization (Lasso)
**Formula:** 
```
Cost = Original Loss + λ Σ|wi|
```
Where λ (lambda) is the regularization parameter and wi are the model weights.

**Key Properties:**
- Adds absolute value of weights to loss function
- Promotes sparsity (drives weights to exactly zero)
- Performs automatic feature selection
- Creates sparse models with fewer non-zero parameters

#### L2 Regularization (Ridge)
**Formula:**
```
Cost = Original Loss + λ Σwi²
```

**Key Properties:**
- Adds squared weights to loss function
- Shrinks weights towards zero but doesn't eliminate them
- Handles multicollinearity well
- Keeps all features but reduces their impact

#### Key Differences

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Sparsity** | Creates sparse models | Keeps all features |
| **Feature Selection** | Automatic | No |
| **Multicollinearity** | Picks one from correlated features | Handles well |
| **Computational** | Not differentiable at 0 | Always differentiable |
| **Geometric Shape** | Diamond (Manhattan distance) | Circle (Euclidean distance) |

#### Effect on Weights - Example

**Before Regularization:**
```python
# Original weights might be:
w = [2.5, -1.8, 0.3, -0.1, 1.9, -0.4]
```

**After L1 Regularization (λ=0.1):**
```python
# Some weights become exactly zero:
w = [2.1, -1.4, 0.0, 0.0, 1.5, 0.0]  # Sparse!
```

**After L2 Regularization (λ=0.1):**
```python
# All weights shrink proportionally:
w = [1.8, -1.3, 0.2, -0.08, 1.4, -0.3]  # All non-zero
```

### Interview-Ready Explanation
"Regularization prevents overfitting by adding a penalty term to the loss function. L1 regularization uses the sum of absolute values of weights, which creates sparse models by driving some weights to exactly zero - this is great for feature selection. L2 regularization uses the sum of squared weights, which shrinks all weights but keeps them non-zero - this handles multicollinearity better and is computationally more stable."

### Common Pitfalls & Follow-ups

**Pitfall 1:** "Regularization always improves model performance"
- **Truth:** It reduces overfitting but might increase bias
- **Follow-up:** "When might regularization hurt performance?"

**Pitfall 2:** "Higher λ always means better regularization"
- **Truth:** Too high λ can cause underfitting
- **Follow-up:** "How do you choose the optimal λ?"

**Pitfall 3:** "L1 is always better for feature selection"
- **Truth:** L1 might miss important correlated features
- **Follow-up:** "What if you have groups of correlated important features?"

---

## 2. Hyperparameter Search for Regularization Parameters

### ELI5 Version
Finding the best regularization parameter is like finding the perfect volume level for your music - too low and you can't hear it (underfitting), too high and it's distorted (overfitting). We try different "volume levels" systematically to find the sweet spot.

### Technical Deep Dive

#### Grid Search
**Concept:** Exhaustively searches through manually specified parameter combinations.

**Implementation:**
```python
# Example parameter grid
param_grid = {
    'lambda': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'alpha': [0.1, 0.5, 0.7, 0.9]  # For Elastic Net
}

# Time Complexity: O(n^p * k * m)
# n = number of values per parameter
# p = number of parameters  
# k = number of CV folds
# m = model training time
```

**Pros:** Guarantees finding global optimum in search space
**Cons:** Exponentially expensive, curse of dimensionality

#### Random Search
**Concept:** Randomly samples parameter combinations from specified distributions.

**Why It Works Better:**
- Most hyperparameters don't affect performance equally
- Random search explores more diverse combinations
- Better budget allocation for important parameters

**Implementation:**
```python
# Random distributions
param_distributions = {
    'lambda': scipy.stats.loguniform(1e-4, 1e2),
    'alpha': scipy.stats.uniform(0, 1)
}

# Typically needs fewer iterations than grid search
```

#### Cross-Validation Strategy

**K-Fold CV Process:**
1. Split data into K folds
2. For each parameter combination:
   - Train on K-1 folds
   - Validate on remaining fold
   - Repeat K times
3. Average validation scores
4. Select best parameters

**Time Series Considerations:**
- Use TimeSeriesSplit instead of random splits
- Maintain temporal order
- Gap between train/validation to prevent data leakage

### Advanced Techniques

#### Bayesian Optimization
- Models the objective function
- Uses acquisition functions to guide search
- More efficient for expensive evaluations
- Tools: Optuna, Hyperopt, scikit-optimize

#### Early Stopping in Search
```python
# Stop if no improvement for n iterations
early_stopping_rounds = 10
no_improvement_count = 0
best_score = -np.inf

for params in search_space:
    score = cross_validate(model, params)
    if score > best_score:
        best_score = score
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= early_stopping_rounds:
            break
```

### Interview-Ready Explanation
"Hyperparameter tuning for regularization typically involves cross-validation to find the optimal λ value. Grid search exhaustively tries all combinations but is computationally expensive. Random search is often more efficient because it explores the space more broadly and most hyperparameters don't equally affect performance. I'd use cross-validation to ensure the chosen parameters generalize well, and consider Bayesian optimization for expensive models."

### Common Pitfalls & Follow-ups

**Pitfall 1:** "Using test set for hyperparameter tuning"
- **Truth:** This causes data leakage and overly optimistic results
- **Follow-up:** "How do you properly split data for hyperparameter tuning?"

**Pitfall 2:** "Not considering computational budget"
- **Truth:** Grid search can be prohibitively expensive
- **Follow-up:** "How would you tune hyperparameters with limited time/compute?"

**Pitfall 3:** "Using wrong CV strategy for time series"
- **Truth:** Random splits break temporal dependencies
- **Follow-up:** "How would you tune hyperparameters for time series forecasting?"

---

## 3. Elastic Net Regularization

### ELI5 Version
Elastic Net is like having both a strict teacher (L1) and a gentle teacher (L2). The strict teacher removes unnecessary things completely, while the gentle teacher makes everything smaller. Elastic Net lets you choose how much of each teaching style to use.

### Technical Deep Dive

#### Formula
```
Cost = Original Loss + λ₁ Σ|wi| + λ₂ Σwi²
Cost = Original Loss + λ * [α * Σ|wi| + (1-α) * Σwi²]
```

Where:
- λ: Overall regularization strength
- α: Mixing parameter (0 ≤ α ≤ 1)
- α = 1: Pure L1 (Lasso)
- α = 0: Pure L2 (Ridge)
- 0 < α < 1: Elastic Net combination

#### Why Elastic Net?

**Problem with Pure L1 (Lasso):**
- In high-dimensional settings (p > n), selects at most n features
- Arbitrary selection among correlated features
- Unstable feature selection with small data changes

**Problem with Pure L2 (Ridge):**
- No feature selection
- Keeps all features with small weights

**Elastic Net Solution:**
- Combines benefits of both
- Stable feature selection
- Handles correlated features better
- Maintains sparsity while managing multicollinearity

#### Mathematical Intuition

**Constraint Regions:**
- L1: Diamond shape (Manhattan ball)
- L2: Circle shape (Euclidean ball)
- Elastic Net: Rounded diamond (convex combination)

The shape allows touching axes (sparsity) while having smooth curves (stability).

#### Practical Parameter Selection

**Common Strategies:**
```python
# Strategy 1: Fix α, tune λ
alpha = 0.5  # Equal weight to L1 and L2
lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]

# Strategy 2: Grid search both
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]

# Strategy 3: Adaptive
# Start with Ridge (α=0), gradually increase α
```

### Real-World Examples

#### Example 1: Gene Expression Analysis
**Problem:** 20,000 genes, 100 patients, highly correlated gene groups
**Solution:** Elastic Net with α=0.7
- **Why:** Need sparsity for interpretation but want to include correlated genes in the same pathway
- **Result:** Selects gene groups rather than individual genes

#### Example 2: Marketing Attribution
**Problem:** Multiple correlated marketing channels, budget allocation
**Solution:** Elastic Net with α=0.5
- **Why:** Want to identify important channels but not completely ignore correlated ones
- **Result:** Balanced feature selection with stable coefficients

#### Example 3: Financial Risk Modeling
**Problem:** Many economic indicators, some highly correlated
**Solution:** Elastic Net with α=0.3 (more Ridge-like)
- **Why:** Need stable predictions, can't afford to ignore important correlated factors
- **Result:** Robust model with controlled overfitting

### Interview-Ready Explanation
"Elastic Net combines L1 and L2 regularization to get the best of both worlds. The α parameter controls the mix - it provides sparsity like L1 but handles correlated features better than pure Lasso. It's particularly useful when you have groups of correlated features that are all relevant, or when p > n and Lasso becomes unstable. The typical approach is to tune both α and λ via cross-validation."

### Common Pitfalls & Follow-ups

**Pitfall 1:** "Elastic Net is always better than L1 or L2"
- **Truth:** Adds complexity and another hyperparameter
- **Follow-up:** "When would you prefer pure L1 or L2 over Elastic Net?"

**Pitfall 2:** "α=0.5 is always the best choice"
- **Truth:** Optimal α depends on data characteristics
- **Follow-up:** "How do you choose α in practice?"

**Pitfall 3:** "Elastic Net solves all multicollinearity problems"
- **Truth:** Still has limitations with extreme correlation
- **Follow-up:** "What other techniques handle multicollinearity?"

---

## 4. Backpropagation

### ELI5 Version
Backpropagation is like learning from mistakes by working backwards. Imagine you're learning to throw a ball into a basket. If you miss, you figure out what went wrong - was your aim too high? Did you throw too hard? Then you adjust each part of your throw step by step, starting from the end result and working backwards to your starting position.

### Technical Deep Dive

#### The Four-Step Process

**Step 1: Forward Pass**
Data flows forward through network layers to produce prediction.

**Step 2: Loss Calculation**
Compare prediction with actual target using loss function.

**Step 3: Backward Pass (Gradient Computation)**
Calculate gradients of loss with respect to each parameter using chain rule.

**Step 4: Weight Update**
Adjust weights using computed gradients and learning rate.

#### Mathematical Foundation

**Chain Rule Application:**
```
∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w
```

Where:
- L: Loss function
- y: Output
- z: Pre-activation
- w: Weight

#### Detailed Example: 2-Layer Network

**Network Architecture:**
```
Input (x) → Hidden Layer (h) → Output Layer (y)
```

**Forward Pass:**
```python
# Layer 1: Input to Hidden
z1 = W1 @ x + b1          # Linear transformation
h = sigmoid(z1)           # Activation function

# Layer 2: Hidden to Output  
z2 = W2 @ h + b2          # Linear transformation
y = sigmoid(z2)           # Final activation

# Loss calculation
L = 0.5 * (y - target)²   # Mean squared error
```

**Backward Pass:**
```python
# Output layer gradients
dL_dy = y - target                    # ∂L/∂y
dy_dz2 = y * (1 - y)                 # ∂y/∂z2 (sigmoid derivative)
dL_dz2 = dL_dy * dy_dz2              # ∂L/∂z2

# Output layer weight/bias gradients
dL_dW2 = dL_dz2 @ h.T                # ∂L/∂W2
dL_db2 = dL_dz2                      # ∂L/∂b2

# Hidden layer gradients (backpropagate error)
dL_dh = W2.T @ dL_dz2                # ∂L/∂h
dh_dz1 = h * (1 - h)                 # ∂h/∂z1 (sigmoid derivative)
dL_dz1 = dL_dh * dh_dz1              # ∂L/∂z1

# Hidden layer weight/bias gradients
dL_dW1 = dL_dz1 @ x.T                # ∂L/∂W1
dL_db1 = dL_dz1                      # ∂L/∂b1
```

**Weight Updates:**
```python
# Gradient descent updates
W2 = W2 - learning_rate * dL_dW2
b2 = b2 - learning_rate * dL_db2
W1 = W1 - learning_rate * dL_dW1
b1 = b1 - learning_rate * dL_db1
```

#### Computational Graph Perspective

**Example: f(x) = (x + 2) * (x - 1)**

```
x = 3
Forward:
  a = x + 2 = 5
  b = x - 1 = 2  
  f = a * b = 10

Backward:
  ∂f/∂a = b = 2
  ∂f/∂b = a = 5
  ∂f/∂x = ∂f/∂a * ∂a/∂x + ∂f/∂b * ∂b/∂x = 2*1 + 5*1 = 7
```

#### Key Implementation Details

**Activation Function Derivatives:**
```python
# Sigmoid: σ(x) = 1/(1 + e^(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ReLU: f(x) = max(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)

# Tanh: f(x) = tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

**Vectorized Implementation:**
```python
class NeuralNetwork:
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]  # batch size
        
        # Output layer
        dz2 = output - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_derivative(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
```

### Interview-Ready Explanation
"Backpropagation is an algorithm for efficiently computing gradients in neural networks using the chain rule. It consists of a forward pass to compute predictions and loss, followed by a backward pass that propagates error gradients from output to input layers. The key insight is that we can reuse intermediate computations from the forward pass to efficiently compute all gradients in one backward traversal. It's essentially automatic differentiation applied to the computational graph of the neural network."

### Common Pitfalls & Follow-ups

**Pitfall 1:** "Gradients are computed layer by layer independently"
- **Truth:** Gradients depend on downstream layers via chain rule
- **Follow-up:** "How does changing a weight in layer 1 affect the final loss?"

**Pitfall 2:** "Backpropagation is just gradient descent"
- **Truth:** Backprop computes gradients; gradient descent uses them for updates
- **Follow-up:** "What's the difference between backpropagation and gradient descent?"

**Pitfall 3:** "Vanishing gradients are a backpropagation problem"
- **Truth:** It's a problem with certain activation functions and architectures
- **Follow-up:** "How do you handle vanishing/exploding gradients?"

**Advanced Follow-ups:**
- "How would you implement backpropagation for a CNN?"
- "What modifications are needed for RNNs?"
- "How does automatic differentiation relate to backpropagation?"

---

## 5. SVD in Linear Regression

### ELI5 Version
Imagine you have a jigsaw puzzle (your data matrix) that's hard to solve directly. SVD is like taking the puzzle apart into three simpler pieces that are much easier to work with. Once you solve the problem with these simpler pieces, you can put them back together to get your answer. It's especially helpful when your puzzle has missing pieces or damaged edges (numerical problems).

### Technical Deep Dive

#### SVD Decomposition
Any matrix A can be decomposed as:
```
A = U Σ V^T
```

Where:
- **U**: Left singular vectors (m×m orthogonal matrix)
- **Σ**: Diagonal matrix of singular values (m×n)
- **V^T**: Right singular vectors (n×n orthogonal matrix)

#### SVD for Linear Regression

**Standard Linear Regression Problem:**
```
min ||Ax - b||²
```

**Normal Equations (potentially unstable):**
```
x = (A^T A)^(-1) A^T b
```

**SVD Solution (numerically stable):**
```
A = U Σ V^T
x = V Σ^(-1) U^T b
```

#### Step-by-Step SVD Regression

**Step 1: Decompose Design Matrix**
```python
# A is n×p design matrix
U, s, Vt = np.linalg.svd(A, full_matrices=False)
# U: n×p, s: p×1, Vt: p×p
```

**Step 2: Transform Target**
```python
# Project target onto left singular vectors
y_transformed = U.T @ y
```

**Step 3: Solve in Transformed Space**
```python
# Divide by singular values (diagonal solve)
beta_transformed = y_transformed / s
```

**Step 4: Transform Back to Original Space**
```python
# Transform back using right singular vectors
beta = Vt.T @ beta_transformed
```

**Complete Implementation:**
```python
def svd_regression(X, y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Handle numerical issues
    tolerance = 1e-10
    s_inv = np.where(s > tolerance, 1/s, 0)
    
    # Compute coefficients
    beta = Vt.T @ (s_inv * (U.T @ y))
    
    return beta
```

#### Advantages of SVD Approach

**1. Numerical Stability**
- Avoids computing A^T A (condition number squared)
- Handles rank-deficient matrices gracefully
- Stable even with ill-conditioned data

**2. Automatic Regularization**
- Small singular values → natural regularization
- Can truncate small singular values for stability

**3. Diagnostics**
- Singular values reveal matrix properties
- Easy to detect multicollinearity
- Condition number = σ_max / σ_min

#### Numerical Stability Example

**Ill-conditioned Problem:**
```python
# Create ill-conditioned matrix
A = np.array([[1, 1], [1, 1.0001], [1, 1.0002]])
b = np.array([2, 3, 4])

# Normal equations (unstable)
try:
    beta_normal = np.linalg.inv(A.T @ A) @ A.T @ b
    print(f"Normal equations: {beta_normal}")
except np.linalg.LinAlgError:
    print("Normal equations failed!")

# SVD solution (stable)
beta_svd = svd_regression(A, b)
print(f"SVD solution: {beta_svd}")

# Check condition numbers
print(f"A condition number: {np.linalg.cond(A)}")
print(f"A^T A condition number: {np.linalg.cond(A.T @ A)}")
```

#### Dimensionality Reduction with SVD

**Truncated SVD for Regression:**
```python
def truncated_svd_regression(X, y, n_components):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Keep only top components
    U_truncated = U[:, :n_components]
    s_truncated = s[:n_components]
    Vt_truncated = Vt[:n_components, :]
    
    # Solve in reduced space
    beta_truncated = Vt_truncated.T @ ((U_truncated.T @ y) / s_truncated)
    
    return beta_truncated
```

**Choosing Number of Components:**
```python
# Method 1: Explained variance
explained_variance_ratio = s**2 / np.sum(s**2)
cumsum_variance = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumsum_variance >= 0.95) + 1

# Method 2: Singular value threshold
threshold = 1e-3 * s[0]  # Relative to largest singular value
n_components = np.sum(s > threshold)

# Method 3: Cross-validation
best_score = -np.inf
for k in range(1, min(X.shape)):
    score = cross_val_score(truncated_svd_regression, X, y, 
                           cv=5, n_components=k)
    if score > best_score:
        best_score = score
        best_k = k
```

#### Relationship to Principal Component Regression

**PCR via SVD:**
```python
def pcr_via_svd(X, y, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # SVD of centered X
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components
    components = Vt[:n_components, :]
    
    # Project X onto principal components
    X_reduced = X_centered @ components.T
    
    # Ordinary least squares in reduced space
    beta_reduced = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
    
    # Transform back to original space
    beta = components.T @ beta_reduced
    
    return beta
```

### Interview-Ready Explanation
"SVD provides a numerically stable way to solve linear regression by decomposing the design matrix into three components: U, Σ, and V^T. Instead of computing the potentially unstable normal equations involving A^T A, SVD allows us to solve the problem in a transformed space where numerical issues are minimized. The singular values in Σ also provide diagnostic information about multicollinearity and can enable dimensionality reduction by truncating small singular values. It's particularly valuable for ill-conditioned or rank-deficient problems."

### Common Pitfalls & Follow-ups

**Pitfall 1:** "SVD is always better than normal equations"
- **Truth:** SVD is more expensive computationally for well-conditioned problems
- **Follow-up:** "When would you prefer normal equations over SVD?"

**Pitfall 2:** "SVD solves regularization automatically"
- **Truth:** It provides some natural regularization but may need explicit control
- **Follow-up:** "How would you add explicit regularization to SVD regression?"

**Pitfall 3:** "All singular values are equally important"
- **Truth:** Small singular values often represent noise or numerical artifacts
- **Follow-up:** "How do you decide which singular values to keep?"

**Advanced Follow-ups:**
- "How does SVD regression relate to ridge regression?"
- "What's the computational complexity compared to normal equations?"
- "How would you handle missing data with SVD?"

---

## 6. Regression Evaluation Metrics

### ELI5 Version
Imagine you're a weather forecaster predicting tomorrow's temperature. Different metrics tell you different things about how good your predictions are:
- **MAE**: On average, how far off are your predictions?
- **MSE**: How far off are you, but being extra harsh on big mistakes?
- **RMSE**: Same as MSE but in the same units as temperature
- **R²**: What percentage of weather patterns can you explain?

### Technical Deep Dive

#### Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n) Σ|yi - ŷi|
```

**Properties:**
- Linear penalty for errors
- Robust to outliers
- Same units as target variable
- Always non-negative

**Interpretation:**
- Average magnitude of prediction errors
- Easy to understand and communicate
- "On average, predictions are off by X units"

**Code Example:**
```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Example
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(f"MAE: {mae(y_true, y_pred)}")  # 0.5
```

#### Mean Squared Error (MSE)

**Formula:**
```
MSE = (1/n) Σ(yi - ŷi)²
```

**Properties:**
- Quadratic penalty (emphasizes large errors)
- Sensitive to outliers
- Units are squared target units
- Always non-negative
- Differentiable everywhere

**Mathematical Significance:**
- Related to variance of residuals
- Optimal under Gaussian noise assumption
- Used in maximum likelihood estimation

**Code Example:**
```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Same example
print(f"MSE: {mse(y_true, y_pred)}")  # 0.375
```

#### Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √(MSE) = √((1/n) Σ(yi - ŷi)²)
```

**Properties:**
- Same units as target variable
- Penalizes large errors more than MAE
- Standard deviation of residuals
- Common in many domains

**Relationship to Standard Deviation:**
```python
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# RMSE ≈ standard deviation of residuals
residuals = y_true - y_pred
print(f"RMSE: {rmse(y_true, y_pred)}")
print(f"Residual std: {np.std(residuals, ddof=1)}")
```

#### R-squared (Coefficient of Determination)

**Formula:**
```
R² = 1 - (SS_res / SS_tot)
R² = 1 - (Σ(yi - ŷi)² / Σ(yi - ȳ)²)
```

Where:
- SS_res: Sum of squares of residuals
- SS_tot: Total sum of squares
- ȳ: Mean of actual values

**Properties:**
- Range: (-∞, 1] (typically [0, 1])
- 1 = perfect prediction
- 0 = model as good as mean
- Negative = worse than mean

**Interpretation:**
- Proportion of variance explained by model
- "Model explains X% of the variance"

**Code Example:**
```python
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Alternative calculation
def r_squared_alt(y_true, y_pred):
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation = correlation_matrix[0, 1]
    return correlation**2
```

#### Adjusted R-squared

**Formula:**
```
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]
```

Where:
- n: number of observations
- p: number of predictors

**Purpose:**
- Adjusts for number of predictors
- Penalizes model complexity
- Can decrease when adding irrelevant features
- Better for model comparison

**Code Example:**
```python
def adjusted_r_squared(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adj_r2
```

#### Mean Absolute Percentage Error (MAPE)

**Formula:**
```
MAPE = (100/n) Σ|((yi - ŷi) / yi)|
```

**Properties:**
- Scale-independent (percentage)
- Easy to interpret
- Problems with zero or near-zero values
- Asymmetric (penalizes under-prediction more)

**Use Cases:**
- Business forecasting
- When relative error matters more than absolute
- Comparing models across different scales

**Code Example:**
```python
def mape(y_true, y_pred, epsilon=1e-8):
    # Add small epsilon to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Symmetric MAPE (alternative)
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
```

### When to Use Each Metric

#### Decision Framework

**Use MAE when:**
- Outliers should not heavily influence evaluation
- You want robust, easy-to-interpret metric
- All errors are equally important
- Business cost is linear in error magnitude

**Use MSE/RMSE when:**
- Large errors are disproportionately bad
- You're using it as loss function (differentiable)
- Working under Gaussian assumptions
- Want to match training objective

**Use R² when:**
- Want to know explanatory power
- Comparing models on same dataset
- Need scale-independent metric
- Communicating to non-technical stakeholders

**Use Adjusted R² when:**
- Comparing models with different numbers of features
- Want to penalize model complexity
- Multiple regression analysis

**Use MAPE when:**
- Relative error is more important than absolute
- Comparing performance across different scales
- Business context focuses on percentages
- Values are always positive and reasonably far from zero

#### Practical Example Comparison

```python
# Dataset with outliers
y_true = np.array([10, 20, 30, 40, 100])  # One outlier
y_pred1 = np.array([12, 18, 32, 38, 95])   # Good model
y_pred2 = np.array([15, 25, 25, 35, 50])   # Handles outlier poorly

print("Model 1 (handles outlier well):")
print(f"MAE: {mae(y_true, y_pred1):.2f}")
print(f"RMSE: {rmse(y_true, y_pred1):.2f}")
print(f"R²: {r_squared(y_true, y_pred1):.3f}")

print("\nModel 2 (poor with outlier):")
print(f"MAE: {mae(y_true, y_pred2):.2f}")
print(f"RMSE: {rmse(y_true, y_pred2):.2f}")
print(f"R²: {r_squared(y_true, y_pred2):.3f}")
```

### Interview-Ready Explanation
"The choice of regression metric depends on your specific use case. MAE gives you the average absolute error and is robust to outliers. MSE/RMSE penalizes large errors more heavily and is useful when big mistakes are particularly costly. R² tells you the proportion of variance explained by your model, which is great for understanding explanatory power. Adjusted R² is better for comparing models with different numbers of features. MAPE is useful when you care about relative rather than absolute errors. I'd typically look at multiple metrics to get a complete picture of model performance."

### Common Pitfalls & Follow-ups

**Pitfall 1:** "R² of 0.8 means the model is good"
- **Truth:** Depends on domain and baseline difficulty
- **Follow-up:** "What factors determine if an R² value is good?"

**Pitfall 2:** "RMSE is always better than MAE"
- **Truth:** Depends on error distribution and business context
- **Follow-up:** "When would you prefer MAE over RMSE?"

**Pitfall 3:** "Higher R² always means better model"
- **Truth:** Can lead to overfitting, need adjusted R² or validation
- **Follow-up:** "How can a model have high R² but poor generalization?"

**Advanced Follow-ups:**
- "How would you create a custom evaluation metric for your business problem?"
- "What metrics would you use for time series forecasting?"
- "How do these metrics relate to different loss functions in training?"

---

## 7. Jaccard Similarity

### ELI5 Version
Imagine you and your friend each have a box of crayons. Jaccard similarity tells you how similar your crayon collections are by looking at: "How many crayons do we both have?" divided by "How many different crayons do we have in total?" If you both have exactly the same crayons, the similarity is 1 (perfect match). If you have completely different crayons, it's 0 (no similarity).

### Technical Deep Dive

#### Definition and Formula

**For Sets A and B:**
```
Jaccard Similarity = |A ∩ B| / |A ∪ B|
J(A,B) = |A ∩ B| / (|A| + |B| - |A ∩ B|)
```

**Jaccard Distance:**
```
Jaccard Distance = 1 - Jaccard Similarity
d(A,B) = 1 - J(A,B)
```

#### Properties

**Mathematical Properties:**
- Range: [0, 1]
- Symmetric: J(A,B) = J(B,A)
- J(A,A) = 1 (identity)
- J(∅,∅) = undefined (convention: 0 or 1)
- Triangle inequality holds for Jaccard distance

**Intuitive Properties:**
- Higher values = more similar sets
- Independent of set sizes (relative measure)
- Only cares about presence/absence, not frequency
- Robust to duplicates within sets

#### Implementation Examples

**Basic Set Implementation:**
```python
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity for two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0  # Convention: empty sets are identical
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

# Example
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
print(f"Jaccard similarity: {jaccard_similarity(A, B)}")  # 0.333
```

**Binary Vector Implementation:**
```python
def jaccard_binary(vec1, vec2):
    """Calculate Jaccard for binary vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    
    intersection = np.sum(vec1 * vec2)  # Both 1
    union = np.sum((vec1 + vec2) > 0)   # At least one 1
    
    return intersection / union if union > 0 else 0.0

# Example
v1 = [1, 1, 0, 1, 0]
v2 = [0, 1, 1, 1, 0]
print(f"Jaccard similarity: {jaccard_binary(v1, v2)}")  # 0.5
```

**Text/String Implementation:**
```python
def jaccard_text(text1, text2, n_grams=1):
    """Calculate Jaccard similarity for text using n-grams."""
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    
    ngrams1 = get_ngrams(text1, n_grams)
    ngrams2 = get_ngrams(text2, n_grams)
    
    return jaccard_similarity(ngrams1, ngrams2)

# Example
text1 = "the quick brown fox"
text2 = "the brown fox is quick"
print(f"Jaccard similarity: {jaccard_text(text1, text2)}")  # 0.6
```

#### Real-World Applications

#### Application 1: Recommendation Systems

**Problem:** Find similar users based on items they've purchased/rated.

**Implementation:**
```python
class JaccardRecommender:
    def __init__(self):
        self.user_items = {}
    
    def fit(self, user_item_matrix):
        """user_item_matrix: dict of user_id -> set of item_ids"""
        self.user_items = user_item_matrix
    
    def user_similarity(self, user1, user2):
        items1 = self.user_items.get(user1, set())
        items2 = self.user_items.get(user2, set())
        return jaccard_similarity(items1, items2)
    
    def recommend(self, target_user, n_recommendations=5):
        target_items = self.user_items.get(target_user, set())
        
        # Find most similar users
        similarities = []
        for user_id, items in self.user_items.items():
            if user_id != target_user:
                sim = self.user_similarity(target_user, user_id)
                similarities.append((user_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Collect recommendations from similar users
        recommendations = set()
        for user_id, similarity in similarities:
            if len(recommendations) >= n_recommendations:
                break
            similar_user_items = self.user_items[user_id]
            new_items = similar_user_items - target_items
            recommendations.update(new_items)
        
        return list(recommendations)[:n_recommendations]

# Example usage
user_items = {
    'user1': {'movie1', 'movie2', 'movie3'},
    'user2': {'movie2', 'movie3', 'movie4'},
    'user3': {'movie1', 'movie4', 'movie5'},
    'user4': {'movie2', 'movie4', 'movie6'}
}

recommender = JaccardRecommender()
recommender.fit(user_items)
print(recommender.recommend('user1'))
```

#### Application 2: Document Clustering

**Problem:** Group similar documents based on their content.

**Implementation:**
```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer

def document_jaccard_clustering(documents, n_clusters=3):
    """Cluster documents using Jaccard similarity."""
    
    # Convert documents to binary term-document matrix
    vectorizer = CountVectorizer(binary=True, lowercase=True, 
                               stop_words='english')
    X_binary = vectorizer.fit_transform(documents).toarray()
    
    # Compute pairwise Jaccard distances
    n_docs = len(documents)
    distance_matrix = np.zeros((n_docs, n_docs))
    
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            jaccard_sim = jaccard_binary(X_binary[i], X_binary[j])
            jaccard_dist = 1 - jaccard_sim
            distance_matrix[i, j] = jaccard_dist
            distance_matrix[j, i] = jaccard_dist
    
    # Hierarchical clustering with precomputed distances
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage='average',
        metric='precomputed'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    return labels, distance_matrix

# Example
documents = [
    "machine learning algorithms are powerful",
    "deep learning neural networks work well", 
    "cats and dogs are pets",
    "artificial intelligence and machine learning",
    "puppies and kittens are cute animals"
]

labels, dist_matrix = document_jaccard_clustering(documents)
print(f"Cluster labels: {labels}")
```

#### Application 3: Market Basket Analysis

**Problem:** Find products frequently bought together.

**Implementation:**
```python
def market_basket_jaccard(transactions, min_similarity=0.1):
    """Find product associations using Jaccard similarity."""
    
    # Convert transactions to item sets
    product_sets = {}
    all_products = set()
    
    for transaction_id, products in transactions.items():
        product_sets[transaction_id] = set(products)
        all_products.update(products)
    
    # Find product co-occurrence patterns
    product_similarities = {}
    products_list = list(all_products)
    
    for i, product1 in enumerate(products_list):
        for j, product2 in enumerate(products_list[i+1:], i+1):
            # Find transactions containing each product
            trans_with_p1 = {tid for tid, products in product_sets.items() 
                           if product1 in products}
            trans_with_p2 = {tid for tid, products in product_sets.items() 
                           if product2 in products}
            
            # Calculate Jaccard similarity of transaction sets
            similarity = jaccard_similarity(trans_with_p1, trans_with_p2)
            
            if similarity >= min_similarity:
                product_similarities[(product1, product2)] = similarity
    
    return product_similarities

# Example transactions
transactions = {
    't1': ['bread', 'milk', 'eggs'],
    't2': ['bread', 'butter', 'jam'],
    't3': ['milk', 'eggs', 'cheese'],
    't4': ['bread', 'milk', 'butter'],
    't5': ['eggs', 'cheese', 'milk']
}

associations = market_basket_jaccard(transactions, min_similarity=0.2)
for (p1, p2), sim in sorted(associations.items(), key=lambda x: x[1], reverse=True):
    print(f"{p1} <-> {p2}: {sim:.3f}")
```

#### Advanced: MinHash for Large-Scale Jaccard

**Problem:** Computing exact Jaccard for large sets is expensive O(|A ∪ B|).

**Solution:** MinHash approximation for efficient estimation.

```python
import hashlib
import random

class MinHashJaccard:
    def __init__(self, num_hashes=100):
        self.num_hashes = num_hashes
        self.hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self):
        """Generate random hash functions."""
        random.seed(42)  # For reproducibility
        hash_funcs = []
        for _ in range(self.num_hashes):
            a = random.randint(1, 2**31 - 1)
            b = random.randint(0, 2**31 - 1)
            hash_funcs.append((a, b))
        return hash_funcs
    
    def _hash_string(self, s, a, b):
        """Simple hash function."""
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        return (a * h + b) % (2**31)
    
    def compute_signature(self, item_set):
        """Compute MinHash signature for a set."""
        signature = []
        for a, b in self.hash_functions:
            min_hash = float('inf')
            for item in item_set:
                h = self._hash_string(str(item), a, b)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature
    
    def estimate_jaccard(self, sig1, sig2):
        """Estimate Jaccard similarity from signatures."""
        matches = sum(1 for s1, s2 in zip(sig1, sig2) if s1 == s2)
        return matches / len(sig1)

# Example usage
minhash = MinHashJaccard(num_hashes=200)

set1 = {'apple', 'banana', 'orange', 'grape'}
set2 = {'banana', 'orange', 'kiwi', 'mango'}

# Exact Jaccard
exact_jaccard = jaccard_similarity(set1, set2)

# MinHash approximation
sig1 = minhash.compute_signature(set1)
sig2 = minhash.compute_signature(set2)
approx_jaccard = minhash.estimate_jaccard(sig1, sig2)

print(f"Exact Jaccard: {exact_jaccard:.3f}")
print(f"MinHash estimate: {approx_jaccard:.3f}")
print(f"Error: {abs(exact_jaccard - approx_jaccard):.3f}")
```

### Interview-Ready Explanation
"Jaccard similarity measures the overlap between two sets by dividing the size of their intersection by the size of their union. It's particularly useful for binary data or when you only care about presence/absence rather than frequency. Common applications include recommendation systems for finding similar users, document clustering, and market basket analysis. For large-scale applications, MinHash can efficiently approximate Jaccard similarity. The metric is robust, symmetric, and ranges from 0 to 1, making it intuitive to interpret."

### Common Pitfalls & Follow-ups

**Pitfall 1:** "Jaccard similarity works well for all types of data"
- **Truth:** Best for binary/categorical data, not suitable for continuous variables
- **Follow-up:** "What similarity measures would you use for continuous features?"

**Pitfall 2:** "Jaccard similarity accounts for item importance"
- **Truth:** Treats all items equally, doesn't consider weights or frequency
- **Follow-up:** "How would you modify Jaccard to account for item importance?"

**Pitfall 3:** "High Jaccard similarity always means good recommendations"
- **Truth:** Can suffer from popularity bias and doesn't account for user preferences
- **Follow-up:** "What are the limitations of collaborative filtering with Jaccard?"

**Advanced Follow-ups:**
- "How would you use Jaccard similarity in a streaming/online setting?"
- "What's the relationship between Jaccard and cosine similarity?"
- "How would you handle the scalability issues with large datasets?"
- "Can you explain the theoretical guarantees of MinHash approximation?"

---

## Summary: Key Interview Tips

### 1. **Structure Your Answers**
- Start with intuitive explanation
- Move to technical details
- Give practical examples
- Discuss limitations and alternatives

### 2. **Common Questions Across All Topics**
- "When would you use X vs Y?"
- "What are the computational complexities?"
- "How do you handle edge cases?"
- "What assumptions does this method make?"

### 3. **Demonstrate Practical Knowledge**
- Mention implementation details
- Discuss hyperparameter tuning
- Show awareness of real-world constraints
- Connect to business objectives

### 4. **Show Critical Thinking**
- Acknowledge trade-offs
- Mention when methods fail
- Suggest improvements or alternatives
- Connect concepts to broader ML pipeline

Remember: Interviews test both your technical knowledge and your ability to think through problems systematically. Practice explaining these concepts at different levels of detail!