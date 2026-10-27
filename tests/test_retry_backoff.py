"""
Tests for retry and backoff logic.
"""
from unittest.mock import patch, MagicMock
import pytest
import requests
from src.utils.retry import with_network_retry

def test_network_retry_success_after_failure():
    mock_func = MagicMock()
    mock_func.side_effect = [
        requests.ConnectionError("Temporary failure"),
        requests.Timeout("Timeout"),
        "Success"
    ]
    
    @with_network_retry
    def resilient_call():
        return mock_func()

    # The decorator suppresses the logging output to test out but we just check return
    result = resilient_call()
    assert result == "Success"
    assert mock_func.call_count == 3

def test_network_retry_max_attempts():
    mock_func = MagicMock()
    mock_func.side_effect = requests.ConnectionError("Permanent failure")
    
    @with_network_retry
    def resilient_call():
        return mock_func()

    with pytest.raises(requests.ConnectionError):
        resilient_call()
        
    assert mock_func.call_count == 5  # Based on stop_after_attempt(5)
