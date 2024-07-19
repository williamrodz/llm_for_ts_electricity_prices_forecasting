from utils import calculate_nmse
import numpy as np
# Test cases
def test_nmse():
    # Test case 1: Perfect prediction
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    actual_nmse = calculate_nmse(y_true, y_pred)
    expected_nmse = 0.0
    assert np.isclose(actual_nmse, expected_nmse), f"Test case 1 failed: actual_nmse={actual_nmse}, expected_nmse={expected_nmse}"
    
    # Test case 2: Prediction with some error
    y_pred = np.array([1, 2, 3, 4, 6])
    actual_nmse = calculate_nmse(y_true, y_pred)
    expected_nmse = 1 / np.var(y_true)
    assert np.isclose(actual_nmse, expected_nmse), f"Test case 2 failed: actual_nmse={actual_nmse}, expected_nmse={expected_nmse}"
    
    # Test case 3: Constant true values, non-constant predictions
    y_true = np.array([5, 5, 5, 5, 5])
    y_pred = np.array([4, 5, 6, 5, 4])
    actual_nmse = calculate_nmse(y_true, y_pred)
    expected_nmse = np.mean((y_true - y_pred) ** 2) / np.var(y_true)
    assert np.isclose(actual_nmse, expected_nmse), f"Test case 3 failed: actual_nmse={actual_nmse}, expected_nmse={expected_nmse}"
    
    # Test case 4: Both true values and predictions are constant
    y_true = np.array([5, 5, 5, 5, 5])
    y_pred = np.array([5, 5, 5, 5, 5])
    actual_nmse = calculate_nmse(y_true, y_pred)
    expected_nmse = 0.0
    assert np.isclose(actual_nmse, expected_nmse), f"Test case 4 failed: actual_nmse={actual_nmse}, expected_nmse={expected_nmse}"
    
    # Test case 5: Large difference in values
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([15, 25, 35, 45, 55])
    actual_nmse = calculate_nmse(y_true, y_pred)
    expected_nmse = np.mean((y_true - y_pred) ** 2) / np.var(y_true)
    assert np.isclose(actual_nmse, expected_nmse), f"Test case 5 failed: actual_nmse={actual_nmse}, expected_nmse={expected_nmse}"

    print("All test cases passed!")


# Run the test cases
test_nmse()