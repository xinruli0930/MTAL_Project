import numpy as np

def calculate_pehe(y_true_treated, y_true_control, y_pred_treated, y_pred_control):
    """
    Calculate Precision in Estimation of Heterogeneous Effects (PEHE).
    
    Parameters:
        y_true_treated: Actual outcomes under treatment.
        y_true_control: Actual outcomes under control.
        y_pred_treated: Predicted outcomes under treatment.
        y_pred_control: Predicted outcomes under control.
        
    Returns:
        pehe: float
    """
    # True individual treatment effects
    true_ite = y_true_treated - y_true_control
    
    # Predicted individual treatment effects
    pred_ite = y_pred_treated - y_pred_control
    
    # Calculate PEHE (root mean squared error between true and predicted ITEs)
    pehe = np.sqrt(np.mean((true_ite - pred_ite)**2))
    
    return pehe

def calculate_ate(y_true_treated, y_true_control, y_pred_treated, y_pred_control):
    """
    Calculate Average Treatment Effect (ATE).
    
    Parameters:
        y_true_treated: Actual outcomes under treatment.
        y_true_control: Actual outcomes under control.
        y_pred_treated: Predicted outcomes under treatment.
        y_pred_control: Predicted outcomes under control.
        
    Returns:
        ate_true: True ATE (float)
        ate_pred: Predicted ATE (float)
        ate_error: Absolute error in estimating ATE (float)
    """
    # True ATE
    ate_true = np.mean(y_true_treated - y_true_control)
    
    # Predicted ATE
    ate_pred = np.mean(y_pred_treated - y_pred_control)
    
    # Absolute error
    ate_error = abs(ate_true - ate_pred)
    
    return ate_true, ate_pred, ate_error

# Example usage:
if __name__ == "__main__":
    # Dummy data for demonstration (replace with your actual results)
    y_true_treated = np.array([1.2, 3.4, 2.1, 4.3, 2.9])
    y_true_control = np.array([0.5, 2.8, 1.8, 3.5, 2.0])
    y_pred_treated = np.array([1.0, 3.5, 2.0, 4.1, 3.0])
    y_pred_control = np.array([0.7, 2.5, 1.9, 3.6, 2.1])

    # Calculate PEHE
    pehe = calculate_pehe(y_true_treated, y_true_control, y_pred_treated, y_pred_control)
    print(f"PEHE: {pehe:.4f}")

    # Calculate ATE
    ate_true, ate_pred, ate_error = calculate_ate(y_true_treated, y_true_control, y_pred_treated, y_pred_control)
    print(f"True ATE: {ate_true:.4f}, Predicted ATE: {ate_pred:.4f}, ATE Error: {ate_error:.4f}")
